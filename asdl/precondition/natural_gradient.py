from typing import List, Union, Any

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from ..core import module_wise_assignments, modules_to_assign
from ..matrices import *
from ..symmatrix import SymMatrix
from ..vector import ParamVector
from ..fisher import LOSS_CROSS_ENTROPY, get_fisher_maker, FisherConfig
from .prec_grad_maker import PreconditionedGradientMaker, PreconditioningConfig
import copy

_normalizations = (nn.BatchNorm1d, nn.BatchNorm2d)
_invalid_ema_decay = -1
_invalid_data_size = -1
_module_level_shapes = [SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_SWIFT_KRON, SHAPE_KFE, SHAPE_UNIT_WISE, SHAPE_DIAG, SHAPE_FOOF, SHAPE_BOOB]

__all__ = [
    'NaturalGradientMaker', 'FullNaturalGradientMaker', 'LayerWiseNaturalGradientMaker',
    'KfacGradientMaker', 'EkfacGradientMaker', 'UnitWiseNaturalGradientMaker', 'DiagNaturalGradientMaker', 'EmpNaturalGradientMaker',
    'FOOFGradientMaker', 'BOOBGradientMaker', 'KfacEmpGradientMaker'
]


class NaturalGradientMaker(PreconditionedGradientMaker):
    """
    GradientMaker for calculating the `Natural Gradient <https://ieeexplore.ieee.org/document/6790500>`

    Args:
        model (torch.nn.Module)
    """
    _supported_classes = (nn.Linear, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d,
                          nn.LayerNorm, nn.Embedding)

    def __init__(self, model, config: PreconditioningConfig,
                 fisher_type: str = FISHER_MC, fisher_shape: Union[str, List[Any]] = SHAPE_FULL,
                 loss_type: str = LOSS_CROSS_ENTROPY, scale: float = 1, grad_scale: float = 1,
                 sync_group: dist.ProcessGroup = None, sync_group_ranks: List[int] = None,
                 module_partitions: List[List[nn.Module]] = None,
                 n_mc_samples: int = 1, var: float = 1, seed: int = None,
                 zero_initialization: bool = False, eye_initialization: bool = False,
                 A_Shape: str = 'Full',B_Shape: str = 'Full'):
        from torch.nn.parallel import DistributedDataParallel as DDP
        if isinstance(model, DDP):
            raise TypeError(f'{DDP} is not supported.')
        del DDP
        super().__init__(model, config)
        if isinstance(fisher_shape, str):
            fisher_shape = [fisher_shape]
        if not self.do_accumulate:
            if config.curvature_upd_ratio is not None:
                raise ValueError('curvature_upd_ratio cannot be specified when no curvature accumulation is performed.')

        self.named_modules_for_curvature = []
        self.modules_for_curvature = []
        self.shape_for = {}
        self.eigen_cut_tf=False
        self.zero_initialization = zero_initialization
        self.eye_initialization = eye_initialization
        self.A_Shape = A_Shape
        self.B_Shape = B_Shape

        for name, module, shapes in module_wise_assignments(self.module_dict,
                                                            *fisher_shape,
                                                            named=True):
            if len(shapes) != 1:
                raise ValueError(f'Each module has to be assigned one Fisher shape. '
                                 f'{name} is assigned {len(shapes)} shapes.')
            self.modules_for_curvature.append(module)
            self.named_modules_for_curvature.append((name, module))
            self.shape_for[module] = shapes[0]
            self.shape_for[name] = shapes[0]
        self._named_modules_for = {}

        if module_partitions is not None:
            if not dist.is_initialized():
                raise EnvironmentError('torch.distributed has to be initialized when module_partitions is set.')
            world_size = dist.get_world_size(sync_group)
            if len(module_partitions) != world_size:
                raise ValueError(f'Number of partitions has to be world_size. Got {len(module_partitions)}')
            if any(len(module_partitions[0]) != len(module_partitions[i]) for i in range(1, world_size)):
                raise ValueError(f'Number of members in each partition has to be the same. '
                                 f'Got {[len(module_partitions[i]) for i in range(world_size)]}')
            self.partitioned_modules = [m for partition in module_partitions for m in partition]
            self.num_modules_per_partition = len(module_partitions[0])
        else:
            self.partitioned_modules = []
            self.num_modules_per_partition = None
        self.module_partitions = module_partitions

        fisher_config = FisherConfig(
            fisher_type=fisher_type,
            fisher_shapes=fisher_shape,
            loss_type=loss_type,
            n_mc_samples=n_mc_samples,
            ignore_modules=config.ignore_modules,
            var=var,
            seed=seed,
        )
        self.fisher_maker = get_fisher_maker(model, fisher_config)
        self.fisher_type = fisher_type
        self.fisher_shape = fisher_shape
        self.scale = scale
        self.grad_scale = grad_scale

        if sync_group is not None:
            if sync_group_ranks is None:
                raise ValueError('sync_group_ranks is not set.')
            if sync_group.size() != len(sync_group_ranks):
                raise ValueError(f'sync_group.size() ({sync_group.size()}) does not match '
                                 f'len(sync_group_ranks) ({len(sync_group_ranks)}).')
        self.sync_group = sync_group
        self.sync_group_ranks = sync_group_ranks

        self.curvature_sync_handles = []
        self.grad_sync_handles = []
        self.grads = []
        self.packed_grads = []

    def do_forward_and_backward(self, step=None):
        return not self.do_update_curvature(step)

    def named_modules_for(self, shape):
        if shape not in self._named_modules_for:
            self._named_modules_for[shape] = list(modules_to_assign(self.model,
                                                                    shape,
                                                                    *self.fisher_shape,
                                                                    ignore_modules=self.config.ignore_modules,
                                                                    named=True))
        return self._named_modules_for[shape]

    def modules_for(self, shape):
        return [m for _, m in self.named_modules_for(shape)]

    def parameters_for(self, shape):
        for module in self.modules_for(shape):
            for p in module.parameters():
                if p.requires_grad:
                    yield p

    @property
    def _fisher_attr(self):
        return self.fisher_maker.config.fisher_attr

    def _get_module_fisher(self, module, postfix=None):
        if postfix is None:
            attr = self._fisher_attr
        else:
            attr = f'{self._fisher_attr}_{postfix}'
        fisher = getattr(module, attr, None)
        return fisher

    def _set_module_fisher(self, module, fisher, postfix=None):
        if postfix is None:
            attr = self._fisher_attr
        else:
            attr = f'{self._fisher_attr}_{postfix}'
        setattr(module, attr, fisher)

    def _get_full_fisher(self):
        return self._get_module_fisher(self.model)

    def _get_module_symmatrix(self, module, shape, postfix=None) -> SymMatrix:
        fisher = self._get_module_fisher(module, postfix)
        if fisher is None:
            return None
        if shape in [SHAPE_FULL, SHAPE_LAYER_WISE]:
            return fisher
        elif shape in [SHAPE_KRON, SHAPE_SWIFT_KRON]:
            return fisher.kron
        elif shape == SHAPE_FOOF:
            return fisher.foof
        elif shape == SHAPE_BOOB:
            return fisher.boob
        elif shape == SHAPE_KFE:
            return fisher.kfe
        elif shape == SHAPE_UNIT_WISE:
            return fisher.unit
        elif shape == SHAPE_DIAG:
            return fisher.diag
        else:
            raise ValueError(f'Invalid shape: {shape}.')

    def _scale_fisher(self, scale):
        for shape in _module_level_shapes:
            for module in self.modules_for(shape):
                matrix = self._get_module_symmatrix(module, shape)
                if matrix is not None:
                    matrix.mul_(scale)
        fisher = self._get_full_fisher()
        if fisher is not None:
            fisher.mul_(scale)

    def _eye_fisher(self, scale):
        for shape in _module_level_shapes:
            for module in self.modules_for(shape):
                matrix = self._get_module_symmatrix(module, shape)
                if matrix is not None:
                    mat = (1-scale) * eye_like(matrix.data)
                    matrix.data._add(mat)
        fisher = self._get_full_fisher()
        if fisher is not None:
            mat = (1-scale) * eye_like(fisher.data)
            fisher.data._add(mat)

        def eye_like(tensor):
            return torch.eye(*tensor.size(), out=torch.empty_like(tensor))

    @property
    def do_accumulate(self):
        return self.config.ema_decay != _invalid_ema_decay

    def update_curvature(self):
        config = self.config
        fisher_maker = self.fisher_maker
        scale = self.scale

        ema_decay = config.ema_decay
        step = self.state['step']
        if ema_decay != _invalid_ema_decay:
            if self.zero_initialization or self.eye_initialization or step != 0:
                scale *= ema_decay
                self._scale_fisher(1 - ema_decay)
            if self.eye_initialization and step==0:
                self._eye_fisher(1 - ema_decay)

        self.delegate_forward_and_backward(fisher_maker,
                                           data_size=self.config.data_size,
                                           scale=scale,
                                           accumulate=self.do_accumulate,
                                           calc_loss_grad=True,
                                           calc_inv=not self.do_accumulate,
                                           damping=self.config.damping
                                           )

    def update_preconditioner(self, damping=None, module_name=None, kron=None, zero_curvature=False, partition_aware=False):
        if not self.do_accumulate:
            return

        if kron is None:
            kron = ['A', 'B']
        if damping is None:
            damping = self.config.damping

        for shape in _module_level_shapes:
            for name, module in self.named_modules_for(shape):
                if module_name is not None:
                    if name != module_name:
                        continue
                    if partition_aware and module in self.partitioned_modules:
                        partition_id = self.partitioned_modules.index(module) // self.num_modules_per_partition
                        module_id_in_partition = self.module_partitions[partition_id].index(module)
                        rank_in_group = dist.get_rank(self.sync_group)
                        modified_partition_id = (partition_id + rank_in_group) % len(self.module_partitions)
                        module = self.module_partitions[modified_partition_id][module_id_in_partition]

                matrix = self._get_module_symmatrix(module, shape)
                if matrix is None:
                    continue

                event = f'inv_{shape}'
                if shape in [SHAPE_KRON, SHAPE_SWIFT_KRON]:
                    for A_or_B in kron:
                        event += f'_{A_or_B}'

                if self.is_module_for_inv_and_precondition(module):
                    if shape in [SHAPE_KRON, SHAPE_SWIFT_KRON]:
                        matrix.update_inv(damping, calc_A_inv='A' in kron, calc_B_inv='B' in kron, A_Shape = self.A_Shape, B_Shape = self.B_Shape)
                    else:
                        matrix.update_inv(damping)

                if zero_curvature:
                    with torch.no_grad():
                        if shape in [SHAPE_KRON, SHAPE_SWIFT_KRON]:
                            if 'A' in kron:
                                matrix.A.mul_(0)
                            if 'B' in kron:
                                matrix.B.mul_(0)
                        else:
                            matrix.mul_(0)

                if module_name is not None:
                    break

        fisher = self._get_full_fisher()
        if fisher is not None:
            fisher.update_inv(damping)
            if zero_curvature:
                with torch.no_grad():
                    fisher.mul_(0)

    def precondition(self, vectors: ParamVector = None, grad_scale=None, use_inv=True):
        if grad_scale is None:
            grad_scale = self.grad_scale
        for shape in _module_level_shapes:
            for module in self.modules_for(shape):
                if not self.is_module_for_inv_and_precondition(module):
                    continue
                self._precondition_module(module, shape, vectors, grad_scale=grad_scale, use_inv=use_inv)
        params = [p for p in self.parameters_for(SHAPE_FULL)]
        if len(params) > 0:
            fisher = self._get_full_fisher()
            if fisher is None:
                raise ValueError(f'Fisher of shape {SHAPE_FULL} has not been calculated.')
            if vectors is None:
                vectors = ParamVector(params, [p.grad for p in params])
            if vectors is None:
                raise ValueError('gradient has not been calculated.')
            if grad_scale != 1:
                vectors.mul_(grad_scale)
            fisher.mvp(vectors=vectors, use_inv=use_inv, inplace=True)

    def _precondition_module(self, module, shape=None, vectors: ParamVector = None,
                            vec_weight: torch.Tensor = None, vec_bias: torch.Tensor = None,
                            grad_scale=None, use_inv=True,inplace=True):
        if grad_scale is None:
            grad_scale = self.grad_scale
        if shape is None:
            for s in _module_level_shapes:
                if module in self.modules_for(s):
                    shape = s
                    break
        if vectors is not None:
            vec_weight = vectors.get_vector_by_param(module.weight, None)
            vec_bias = vectors.get_vector_by_param(module.bias, None)
        if shape is None:
            raise ValueError(f'No shape is assigned to module: {module}.')
        matrix = self._get_module_symmatrix(module, shape)
        if matrix is None:
            raise ValueError(f'Matrix of shape {shape} for module {module} has not been calculated.')
        if vec_weight is None and module.weight.requires_grad:
            vec_weight = module.weight.grad
        if vec_weight is None:
            raise ValueError(f'weight gradient for module {module} has not been calculated.')
        if _bias_requires_grad(module):
            if vec_bias is None:
                vec_bias = module.bias.grad
            if vec_bias is None:
                raise ValueError(f'bias gradient for module {module} has not been calculated.')
        if grad_scale != 1:
            vec_weight.data.mul_(grad_scale)
            if vec_bias is not None:
                vec_bias.data.mul_(grad_scale)
        if not use_inv or matrix.has_inv:
            kwargs = dict(vec_weight=vec_weight, vec_bias=vec_bias, use_inv=use_inv, inplace=inplace)
            if shape == SHAPE_KFE:
                kwargs['eps'] = self.config.damping
            return matrix.mvp(**kwargs)
        if vec_bias is not None:
            return vec_weight,vec_bias
        else:
            return vec_weight
    
    def vector_precond(self,module,vec,vec_bias=None,inv=False):
        for s in _module_level_shapes:
            if module in self.modules_for(s):
                shape = s
                break
        if vec_bias is not None:
            pre_G,pre_Gias=self._precondition_module(module,shape=shape,vec_weight=vec,vec_bias=vec_bias,inplace=False,use_inv=not inv)
            pre_G = torch.cat([pre_G, pre_Gias.unsqueeze(-1)], dim=1)
        else:
            pre_G=self._precondition_module(module,shape=shape,vec_weight=vec,inplace=False, use_inv = not inv)
        return pre_G

    def is_module_for_inv_and_precondition(self, module: nn.Module):
        if module not in self.modules_for_curvature:
            return False
        module_partitions = self.module_partitions
        if module_partitions is None:
            return True
        if module not in self.partitioned_modules:
            return True
        else:
            rank = dist.get_rank(self.sync_group)
            return module in module_partitions[rank]

    def sync_curvature(self, module_name=None, kron=None, diag=None, with_grad=False, enabled=True, async_op=False):
        if not enabled:
            return
        handles = []
        if self.module_partitions is not None:
            if module_name is not None:
                handles += self.reduce_curvature(module_name, kron=kron, diag=diag, with_grad=with_grad)
            else:
                handles += self.reduce_scatter_curvature(kron=kron, diag=diag, with_grad=with_grad)
        handles += self.all_reduce_undivided_curvature(module_name=module_name, kron=kron, diag=diag, with_grad=with_grad)
        if async_op:
            self.curvature_sync_handles += handles
        else:
            for handle in handles:
                handle.wait()

    def sync_grad_pre_precondition(self, enabled=True, async_op=False):
        if not enabled:
            return
        if self.module_partitions is not None:
            self.reduce_scatter_grad(async_op=async_op)
        self.all_reduce_undivided_grad(async_op=async_op)

    def sync_grad_post_precondition(self, enabled=True, async_op=False):
        if not enabled:
            return
        if self.module_partitions is not None:
            self.all_gather_grad(async_op=async_op)
        self.all_reduce_no_curvature_grad(async_op=async_op)

    def reduce_scatter_curvature(self, kron=None, diag=None, with_grad=False):
        module_partitions = self.module_partitions
        if module_partitions is None:
            raise ValueError('module_partitions is not set.')
        handles = []
        for shape in _module_level_shapes:
            keys_list = self._keys_list_from_shape(shape, kron=kron, diag=diag)
            for keys in keys_list:
                handles += self.fisher_maker.reduce_scatter_fisher(module_partitions,
                                                                   *keys,
                                                                   with_grad=with_grad,
                                                                   group=self.sync_group,
                                                                   async_op=True)
        return handles

    def reduce_curvature(self, module_name, kron=None, diag=None, with_grad=False):
        module_partitions = self.module_partitions
        if module_partitions is None:
            raise ValueError('module_partitions is not set.')
        try:
            module = next(m for name, m in self.named_modules_for_curvature if name == module_name)
            if module not in self.partitioned_modules:
                return []
            dst = next(i for i, partition in enumerate(module_partitions) if module in partition)
            if self.sync_group is not None:
                dst = self.sync_group_ranks[dst]
        except StopIteration:
            return []
        keys_list = self._keys_list_from_shape(self.shape_for[module], kron=kron, diag=diag)
        handles = []
        for keys in keys_list:
            handles += self.fisher_maker.reduce_fisher([module],
                                                       *keys,
                                                       all_reduce=False,
                                                       dst=dst,
                                                       with_grad=with_grad,
                                                       group=self.sync_group,
                                                       async_op=True)
        return handles

    def all_reduce_undivided_curvature(self, module_name=None, kron=None, diag=None, with_grad=False):
        modules = []
        for name, module in self.named_modules_for_curvature:
            if module in self.partitioned_modules:
                continue
            if module_name is not None and name != module_name:
                continue
            modules.append(module)
        handles = []
        for shape in _module_level_shapes:
            keys_list = self._keys_list_from_shape(shape, kron=kron, diag=diag)
            for keys in keys_list:
                handles += self.fisher_maker.reduce_fisher(modules,
                                                           *keys,
                                                           all_reduce=True,
                                                           with_grad=with_grad,
                                                           group=self.sync_group,
                                                           async_op=True)
        return handles

    @staticmethod
    def _keys_list_from_shape(shape, kron=None, diag=None):
        if shape == SHAPE_FULL:
            return [['data']]
        elif shape == SHAPE_LAYER_WISE:
            return [['data']]
        elif shape in [SHAPE_KRON, SHAPE_SWIFT_KRON]:
            if kron is None:
                kron = ['A', 'B']
            if any(A_or_B not in ['A', 'B'] for A_or_B in kron):
                raise ValueError(f'kron has to be a list of "A" or "B". Got {kron}')
            return [['kron', A_or_B] for A_or_B in kron]
        elif shape == SHAPE_UNIT_WISE:
            return [['unit', 'data']]
        elif shape == SHAPE_DIAG:
            if diag is None:
                diag = ['weight', 'bias']
            if any(w_or_b not in ['weight', 'bias'] for w_or_b in diag):
                raise ValueError(f'diag has to be a list of "weight" or "bias". Got {diag}')
            return [['diag', w_or_b] for w_or_b in diag]

    def reduce_scatter_grad(self, async_op=False):
        self._scatter_or_gather_grad('scatter', async_op=async_op)

    def all_gather_grad(self, async_op=False):
        self._scatter_or_gather_grad('gather', async_op=async_op)

    def _scatter_or_gather_grad(self, scatter_or_gather, async_op=False):
        if not dist.is_initialized():
            raise EnvironmentError('torch.distributed is not initialized.')
        group = self.sync_group
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        module_partitions = self.module_partitions
        if module_partitions is None:
            raise ValueError('module_partitions is not set.')
        if len(module_partitions) != world_size:
            raise ValueError(f'Number of partitions has to be world_size. Got {len(module_partitions)}')
        if any(len(module_partitions[0]) != len(module_partitions[i]) for i in range(1, world_size)):
            raise ValueError(f'Number of members in each partition has to be the same. '
                             f'Got {[len(module_partitions[i]) for i in range(world_size)]}')
        num_modules_per_partition = len(module_partitions[0])
        for i in range(num_modules_per_partition):
            tensor_list = []
            grads_list = []
            for j in range(world_size):
                grads = [p.grad for p in module_partitions[j][i].parameters() if p.requires_grad and p.grad is not None]
                grads_list.append(grads)
                tensor_list.append(parameters_to_vector(grads))
            if scatter_or_gather == 'scatter':
                handle = dist.reduce_scatter(tensor_list[rank], tensor_list, group=group, async_op=async_op)
                if async_op:
                    self.grad_sync_handles.append(handle)
                    self.grads.append(grads_list[rank])
                    self.packed_grads.append(tensor_list[rank])
                else:
                    vector_to_parameters(tensor_list[rank], grads_list[rank])
            else:
                handle = dist.all_gather(tensor_list, tensor_list[rank], group=group, async_op=async_op)
                if async_op:
                    self.grad_sync_handles.append(handle)
                    self.grads.append([grads_list[j] for j in range(world_size)])
                    self.packed_grads.append([tensor_list[j] for j in range(world_size)])
                else:
                    for j in range(world_size):
                        vector_to_parameters(tensor_list[j], grads_list[j])

    def all_reduce_undivided_grad(self, async_op=False):
        if not dist.is_initialized():
            raise EnvironmentError('torch.distributed is not initialized.')
        module_list = nn.ModuleList([m for m in self.modules_for_curvature if m not in self.partitioned_modules])
        self._all_reduce_grad(module_list, async_op=async_op)

    def all_reduce_no_curvature_grad(self, async_op=False):
        module_list = nn.ModuleList([m for m in self.model.modules()
                                     if len(list(m.children())) == 0 and m not in self.modules_for_curvature])
        self._all_reduce_grad(module_list, async_op=async_op)

    def _all_reduce_grad(self, module: nn.Module, async_op=False):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if len(grads) == 0:
            return
        packed_tensor = parameters_to_vector(grads)
        handle = dist.all_reduce(packed_tensor, group=self.sync_group, async_op=async_op)
        if async_op:
            self.grad_sync_handles.append(handle)
            self.grads.append(grads)
            self.packed_grads.append(packed_tensor)
        else:
            vector_to_parameters(packed_tensor, grads)

    def wait_all_curvature_sync(self):
        for _ in range(len(self.curvature_sync_handles)):
            self.curvature_sync_handles.pop(0).wait()

    def wait_all_grad_sync(self):
        for _ in range(len(self.grad_sync_handles)):
            self.grad_sync_handles.pop(0).wait()
            grads = self.grads.pop(0)
            packed_grads = self.packed_grads.pop(0)
            if isinstance(grads, list) and isinstance(grads[0], list):
                if not isinstance(packed_grads, list):
                    raise TypeError(f'packed_grads has to be list. Got {type(packed_grads)}.')
                for p, g in zip(packed_grads, grads):
                    vector_to_parameters(p, g)
            else:
                vector_to_parameters(packed_grads, grads)

    def remove_damping(self):
        for module in self.module_dict.children():
            for s in _module_level_shapes:
                if module in self.modules_for(s):
                    shape = s
                    break
            matrix = self._get_module_symmatrix(module, shape)

            if hasattr(matrix,'data') and not hasattr(matrix,'A'):
                A = matrix.data
                diagA = torch.diagonal(A)
                diagA -= self.config.damping
                matrix.data = A

            if hasattr(matrix,'A'):
                A = matrix.A
                diagA = torch.diagonal(A)
                diagA -= matrix.damping_A
                matrix.A = A

            if hasattr(matrix,'B'):
                B = matrix.B
                diagB = torch.diagonal(B)
                diagB -= matrix.damping_B
                matrix.B = B

    def add_damping(self):
        for module in self.module_dict.children():
            for s in _module_level_shapes:
                if module in self.modules_for(s):
                    shape = s
                    break
            matrix = self._get_module_symmatrix(module, shape)
            if hasattr(matrix,'data') and not hasattr(matrix,'A'):
                A = matrix.data
                diagA = torch.diagonal(A)
                diagA += self.config.damping
                matrix.data = A

            if hasattr(matrix,'A'):
                A = matrix.A
                diagA = torch.diagonal(A)
                diagA += matrix.damping_A
                matrix.A = A

            if hasattr(matrix,'B'):
                B = matrix.B
                diagB = torch.diagonal(B)
                diagB += matrix.damping_B
                matrix.B = B

    def substantive_damping(self,vs,Fvs):
        params = list(self.module_dict.parameters())
        vs = [torch.randn_like(p) for p in params]
        vs2 = ParamVector(params,vs)
        f_maker = self.fisher_maker
        f_maker.config.data_size=self.config.data_size
        Pvs = ParamVector(params,vs).copy()
        self.add_damping()
        self.precondition(vectors = Pvs,use_inv=False)
        self.remove_damping()
        damping = Pvs.add(Fvs,alpha=-1).norm()/vs2.norm()
        return damping

    def fetch_cholesky(self,module):
        for s in _module_level_shapes:
            if module in self.modules_for(s):
                shape = s
                break
        matrix = self._get_module_symmatrix(module, shape)
        return matrix.A,matrix.B

    def fetch_eigen_range(self,module):
        for s in _module_level_shapes:
            if module in self.modules_for(s):
                shape = s
                break
        matrix = self._get_module_symmatrix(module, shape)

        if SHAPE_FOOF in self.fisher_shape:
            A=matrix.A
            A = self.damp_matrix(matrix=A,damping=matrix.damping_A)
            LA = torch.linalg.eigvalsh(A)
            max_eigen=float(torch.max(LA))
            min_eigen=float(torch.min(LA))
        else:
            A,B=matrix.A,matrix.B
            A = self.damp_matrix(matrix=A,damping=matrix.damping_A)
            B = self.damp_matrix(matrix=B,damping=matrix.damping_B)
            LA = torch.linalg.eigvalsh(A)
            LB = torch.linalg.eigvalsh(B)
            max_eigen=float(torch.max(LA)*torch.max(LB))
            min_eigen=float(torch.min(LA)*torch.min(LB))

        return max_eigen,min_eigen

    def fetch_ABeigen_range(self,module,dmp=True):
        for s in _module_level_shapes:
            if module in self.modules_for(s):
                shape = s
                break
        matrix = self._get_module_symmatrix(module, shape)

        if SHAPE_FOOF in self.fisher_shape:
            A=matrix.A
            if dmp:
                A = self.damp_matrix(matrix=A,damping=matrix.damping_A)
            LA = torch.linalg.eigvalsh(A)
            A_max_eigen = torch.max(LA)
            B_max_eigen = 1
        else:
            A,B=matrix.A,matrix.B
            if dmp:
                A = self.damp_matrix(matrix=A,damping=matrix.damping_A)
                B = self.damp_matrix(matrix=B,damping=matrix.damping_B)
            LA = torch.linalg.eigvalsh(A)
            LB = torch.linalg.eigvalsh(B)
            A_max_eigen = torch.max(LA)
            B_max_eigen = torch.max(LB)
        
        return A_max_eigen,B_max_eigen

    def fetch_trace(self,module):
        for s in _module_level_shapes:
            if module in self.modules_for(s):
                shape = s
                break
        matrix = self._get_module_symmatrix(module, shape)

        if SHAPE_FOOF in self.fisher_shape:
            A=matrix.A
            trace_A = torch.trace(A)
            trace_B = 0
        else:
            A,B=matrix.A,matrix.B
            trace_A = torch.trace(A)
            trace_B = torch.trace(B)
        return trace_A,trace_B
    
    def fetch_diagnorm(self,module,dmp=True):
        for s in _module_level_shapes:
            if module in self.modules_for(s):
                shape = s
                break
        matrix = self._get_module_symmatrix(module, shape)

        if SHAPE_FOOF in self.fisher_shape:
            A=matrix.A
            if dmp:
                A = self.damp_matrix(matrix=A,damping=matrix.damping_A)
            A_ratio = torch.norm(torch.diag(A))/torch.norm(A)
            B_ratio = 1
        else:
            A,B=matrix.A,matrix.B
            if dmp:
                A = self.damp_matrix(matrix=A,damping=matrix.damping_A)
                B = self.damp_matrix(matrix=B,damping=matrix.damping_B)
            A_ratio = torch.norm(torch.diag(A))/torch.norm(A)
            B_ratio = torch.norm(torch.diag(B))/torch.norm(B)

        return A_ratio,B_ratio
    
    def fetch_all_diagnorm(self,dmp=True):
        A_diag_norm=0
        B_diag_norm=0
        A_norm=0
        B_norm=0

        for name, module in self.module_dict.items():
            for s in _module_level_shapes:
                if module in self.modules_for(s):
                    shape = s
                    break
            matrix = self._get_module_symmatrix(module, shape)

            if SHAPE_FOOF in self.fisher_shape:
                A=matrix.A
                if dmp:
                    A = self.damp_matrix(matrix=A,damping=matrix.damping_A)
                A_diag_norm += torch.norm(torch.diag(A))**2
                A_norm += torch.norm(A)**2
                B_diag_norm += 1
                B_norm += 1
            else:
                A,B=matrix.A,matrix.B
                if dmp:
                    A = self.damp_matrix(matrix=A,damping=matrix.damping_A)
                    B = self.damp_matrix(matrix=B,damping=matrix.damping_B)
                A_diag_norm += torch.norm(torch.diag(A))**2
                A_norm += torch.norm(A)**2
                B_diag_norm += torch.norm(torch.diag(B))**2
                B_norm += torch.norm(B)**2

        return (A_diag_norm/A_norm)**0.5,(B_diag_norm/B_norm)**0.5

    def fetch_damping(self,module):
        for s in _module_level_shapes:
            if module in self.modules_for(s):
                shape = s
                break
        matrix = self._get_module_symmatrix(module, shape)

        return matrix.damping_A,matrix.damping_B

    def fetch_eigen_mean(self,module):
        for s in _module_level_shapes:
            if module in self.modules_for(s):
                shape = s
                break
        matrix = self._get_module_symmatrix(module, shape)

        if SHAPE_FOOF in self.fisher_shape:
            NotImplemented
        else:
            A,B=matrix.A,matrix.B
            trace_A = torch.trace(A)
            trace_B = torch.trace(B)
        return (trace_A*trace_B)/(A.size()[0]*B.size()[1]),trace_A/A.size()[0],trace_B/B.size()[1]

    def damp_matrix(self,matrix,damping):
        M = matrix.detach().clone()
        diagA = torch.diagonal(M)
        diagA += damping
        return M

class FullNaturalGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: PreconditioningConfig, *args, **kwargs):
        super().__init__(model, config, *args, **kwargs, fisher_shape=SHAPE_FULL)


class LayerWiseNaturalGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: PreconditioningConfig, *args, **kwargs):
        super().__init__(model, config, *args, **kwargs, fisher_shape=SHAPE_LAYER_WISE)


class KfacGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: PreconditioningConfig, *args, swift=False, **kwargs):
        fisher_shape = [SHAPE_SWIFT_KRON if swift else SHAPE_KRON,
                        (nn.BatchNorm1d, SHAPE_UNIT_WISE),
                        (nn.BatchNorm2d, SHAPE_UNIT_WISE),
                        (nn.LayerNorm, SHAPE_UNIT_WISE)]
        super().__init__(model, config, *args, **kwargs, fisher_shape=fisher_shape)

class KfacEmpGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: PreconditioningConfig, *args, swift=False, **kwargs):
        fisher_shape = [SHAPE_SWIFT_KRON if swift else SHAPE_KRON,
                        (nn.BatchNorm1d, SHAPE_UNIT_WISE),
                        (nn.BatchNorm2d, SHAPE_UNIT_WISE),
                        (nn.LayerNorm, SHAPE_UNIT_WISE)]
        super().__init__(model, config, *args, **kwargs, fisher_shape=fisher_shape,fisher_type=FISHER_EMP)


class EkfacGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: PreconditioningConfig, *args, **kwargs):
        super().__init__(model, config, *args, **kwargs, fisher_shape=SHAPE_KFE)
        if self.fisher_type != FISHER_EMP:
            raise ValueError(f'{EkfacGradientMaker} supports only {FISHER_EMP}.')

    def _update_preconditioner(self, *args, **kwargs):
        pass

    def _precondition(self, vectors: ParamVector = None, grad_scale=None, use_inv=False):
        if use_inv:
            raise ValueError('EKFAC does not calculate the inverse matrix.')
        super().precondition(vectors=vectors, grad_scale=grad_scale, use_inv=False)


class UnitWiseNaturalGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: PreconditioningConfig, *args, **kwargs):
        super().__init__(model, config, *args, **kwargs, fisher_shape=SHAPE_UNIT_WISE)


class DiagNaturalGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: PreconditioningConfig, *args, **kwargs):
        super().__init__(model, config, *args, **kwargs, fisher_shape=SHAPE_DIAG)


class EmpNaturalGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: PreconditioningConfig, *args, **kwargs):
        super().__init__(model, config, *args, **kwargs, fisher_type=FISHER_EMP)

class FOOFGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: PreconditioningConfig, *args, **kwargs):
        super().__init__(model, config, *args, **kwargs, fisher_shape=SHAPE_FOOF)

class BOOBGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: PreconditioningConfig, *args, **kwargs):
        super().__init__(model, config, *args, **kwargs, fisher_shape=SHAPE_BOOB)

def _bias_requires_grad(module):
    return hasattr(module, 'bias') \
           and module.bias is not None \
           and module.bias.requires_grad
