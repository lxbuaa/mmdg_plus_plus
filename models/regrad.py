import torch
import re


def EU_dist(x1, x2):
    d_matrix = torch.zeros(x1.shape[0], x2.shape[0]).to(x1.device)
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            d = torch.sqrt(torch.dot((x1[i] - x2[j]), (x1[i] - x2[j])))
            d_matrix[i, j] = d
    return d_matrix


def get_layer_id(l_name):
    match = re.search(r'\d+', l_name.split('.')[0])
    if match:
        first_number = match.group()
        return first_number
    else:
        return None


def cal_orthogonal_grad(base_grad, decompose_grad, eps=1e-8):
    """
    :param base_grad: 基grad
    :param decompose_grad: 要分解的grad
    :param eps: 防止除出nan
    :return:
    """
    grad_norm = torch.dot(base_grad.flatten(), base_grad.flatten()) + eps
    proj_len = torch.dot(base_grad.flatten(), decompose_grad.flatten())
    factor = proj_len / grad_norm
    factor = torch.tensor(0.0) if torch.isnan(factor) else factor
    non_orthogonal_grad = factor * base_grad
    return non_orthogonal_grad


def cal_same_dir_grad(base_grad, decompose_grad, eps=1e-8):
    """
    获取decompose_grad中与base_grad同向的部分
    :param base_grad: 基grad
    :param decompose_grad: 要分解的grad
    :param eps: 防止除出nan
    :return:
    """
    grad_norm = torch.dot(base_grad.flatten(), base_grad.flatten()) + eps
    proj_len = torch.dot(base_grad.flatten(), decompose_grad.flatten())
    factor = proj_len / grad_norm
    factor = torch.tensor(0.0) if torch.isnan(factor) else factor
    factor = factor if factor >= 0.0 else torch.tensor(0.0)
    non_orthogonal_grad = factor * base_grad

    return non_orthogonal_grad


def delete_conflict_grad(base_grad, decompose_grad, eps=1e-8):
    """
    获取decompose_grad中与base_grad同向的部分
    :param base_grad: 基grad
    :param decompose_grad: 要分解的grad
    :param eps: 防止除出nan
    :return:
    """
    grad_norm = torch.dot(base_grad.flatten(), base_grad.flatten()) + eps
    proj_len = torch.dot(base_grad.flatten(), decompose_grad.flatten())
    factor = proj_len / grad_norm
    factor = torch.tensor(0.0) if torch.isnan(factor) else factor
    orthogonal_grad = decompose_grad - base_grad * factor
    factor = factor if factor >= 0.0 else torch.tensor(0.0)
    non_orthogonal_grad = factor * base_grad

    return orthogonal_grad + non_orthogonal_grad


def get_slow_modal_grad(main_grad, sub_grad_1, sub_grad_2, version=4):
    """
    调制慢模态
    :param main_grad:
    :param sub_grad_1:
    :param sub_grad_2:
    :return:
    """
    if version == 1:
        if sub_grad_1 is not None:
            """去除grad1在main_grad上的非垂直部分"""
            sub_grad_1_non_orthogonal = cal_orthogonal_grad(main_grad, sub_grad_1)
            sub_grad_1_orthogonal = (sub_grad_1 - sub_grad_1_non_orthogonal)
        else:
            sub_grad_1_orthogonal = 0.0

        if sub_grad_2 is not None:
            """去除grad2在main_grad上的非垂直部分"""
            sub_grad_2_non_orthogonal = cal_orthogonal_grad(main_grad, sub_grad_2)
            sub_grad_2_orthogonal = (sub_grad_2 - sub_grad_2_non_orthogonal)
        else:
            sub_grad_2_orthogonal = 0.0
        return main_grad, sub_grad_1_orthogonal, sub_grad_2_orthogonal
    else:
        sub_grad_1 = sub_grad_1 if sub_grad_1 is not None else torch.zeros_like(main_grad)
        sub_grad_2 = sub_grad_2 if sub_grad_2 is not None else torch.zeros_like(main_grad)
        if version == 2:
            """只保留方向完全相同的"""
            modulated_sub_grad_1 = cal_same_dir_grad(main_grad, sub_grad_1)
            modulated_sub_grad_2 = cal_same_dir_grad(main_grad, sub_grad_2)
        elif version == 3:
            """只删除方向完全相反的"""
            modulated_sub_grad_1 = delete_conflict_grad(main_grad, sub_grad_1)
            modulated_sub_grad_2 = delete_conflict_grad(main_grad, sub_grad_2)
        elif version == 4:
            """结合前两种，夹角大于90度使用3，夹角小于90度使用2"""
            if torch.dot(main_grad.flatten(), sub_grad_1.flatten()) > 0.0:
                modulated_sub_grad_1 = cal_same_dir_grad(main_grad, sub_grad_1)
            else:
                modulated_sub_grad_1 = delete_conflict_grad(main_grad, sub_grad_1)

            if torch.dot(main_grad.flatten(), sub_grad_2.flatten()) > 0.0:
                modulated_sub_grad_2 = cal_same_dir_grad(main_grad, sub_grad_2)
            else:
                modulated_sub_grad_2 = delete_conflict_grad(main_grad, sub_grad_2)

        return main_grad, modulated_sub_grad_1, modulated_sub_grad_2


def get_fast_modal_grad(main_grad, sub_grad_1, sub_grad_2, version=4):
    """
    调制快模态
    :param main_grad:
    :param sub_grad_1:
    :param sub_grad_2:
    :return:
    """
    if version == 1:
        main_grad_1 = 0.5 * main_grad
        main_grad_1_orthogonal = 0.5 * main_grad
        main_grad_2 = 0.5 * main_grad
        main_grad_2_orthogonal = 0.5 * main_grad

        if sub_grad_1 is None:
            main_grad_1 = 0.0
            main_grad_1_orthogonal = 0.0
            main_grad_2 = main_grad
            main_grad_2_orthogonal = main_grad

        if sub_grad_2 is None:
            main_grad_2 = 0.0
            main_grad_2_orthogonal = 0.0
            main_grad_1 = main_grad
            main_grad_1_orthogonal = main_grad

        if sub_grad_1 is not None:
            """去除main_grad在grad_1上的非垂直部分"""
            main_grad_1_non_orthogonal = cal_orthogonal_grad(sub_grad_1, main_grad_1)
            main_grad_1_orthogonal = main_grad_1 - main_grad_1_non_orthogonal
        else:
            sub_grad_1 = 0.0

        if sub_grad_2 is not None:
            """去除main_grad在grad_2上的非垂直部分"""
            main_grad_2_non_orthogonal = cal_orthogonal_grad(sub_grad_2, main_grad_2)
            main_grad_2_orthogonal = main_grad_2 - main_grad_2_non_orthogonal
        else:
            sub_grad_2 = 0.0

        return main_grad_1_orthogonal + main_grad_2_orthogonal, sub_grad_1, sub_grad_2
    else:
        sub_grad_1 = sub_grad_1 if sub_grad_1 is not None else torch.zeros_like(main_grad)
        sub_grad_2 = sub_grad_2 if sub_grad_2 is not None else torch.zeros_like(main_grad)
        sub_grad_sum = sub_grad_1 + sub_grad_2

        if version == 2:
            """只保留方向完全相同的"""
            modulated_main_grad = cal_same_dir_grad(sub_grad_sum, main_grad)
        elif version == 3:
            """只去掉方向完全相反的"""
            modulated_main_grad = delete_conflict_grad(sub_grad_sum, main_grad)
        elif version == 4:
            """结合前两种，夹角大于90度使用3，夹角小于90度使用2"""
            if torch.dot(main_grad.flatten(), sub_grad_sum.flatten()) > 0.0:
                modulated_main_grad = cal_same_dir_grad(sub_grad_sum, main_grad)
            else:
                modulated_main_grad = delete_conflict_grad(sub_grad_sum, main_grad)

        return modulated_main_grad, sub_grad_1, sub_grad_2


def get_named_parameters_with_grad(model, get_type='name'):
    """
    获取参数/参数的梯度列表，避免拿到没有grad的class_token
    :param model:
    :param get_type:
    :return:
    """
    named_param_list = []
    for layer_name, param in model.named_parameters():
        # if hasattr(param, 'grad_clone'):
        if get_type == 'name':
            named_param_list.append(layer_name)
        elif get_type == 'grad':
            if param.grad is not None:
                named_param_list.append(param.grad.clone())
                # print(layer_name, 'Not None')
            else:
                named_param_list.append(None)

    return named_param_list


class ReGrad:
    def __init__(self, momentum_factor=0.5):
        self.mont_loss_1 = 0.0
        self.mont_loss_2 = 0.0
        self.mont_loss_3 = 0.0
        self.mont_factor = momentum_factor

    def cal_loss_momentum(self, model, optimizer, loss_dict, version=4):
        loss_1 = loss_dict['m1']
        loss_2 = loss_dict['m2']
        loss_3 = loss_dict['m3']
        loss_total = loss_dict['total']
        self.mont_loss_1 += self.mont_loss_1 * self.mont_factor + loss_1.item() * (1.0 - self.mont_factor)
        self.mont_loss_2 += self.mont_loss_2 * self.mont_factor + loss_2.item() * (1.0 - self.mont_factor)
        self.mont_loss_3 += self.mont_loss_3 * self.mont_factor + loss_3.item() * (1.0 - self.mont_factor)

        temp_loss_dict = {
            "total": torch.tensor(loss_total.item()),
            "m1": torch.tensor(loss_1.item()),
            "m2": torch.tensor(loss_2.item()),
            "m3": torch.tensor(loss_3.item()),
        }

        optimizer.zero_grad()
        loss_1.backward(retain_graph=True)
        grad_l1_list = get_named_parameters_with_grad(model, 'grad')

        optimizer.zero_grad()
        loss_2.backward(retain_graph=True)
        grad_l2_list = get_named_parameters_with_grad(model, 'grad')

        optimizer.zero_grad()
        loss_3.backward(retain_graph=True)
        grad_l3_list = get_named_parameters_with_grad(model, 'grad')

        optimizer.zero_grad()
        loss_total.backward(retain_graph=False)  # 最后一个loss，不需要维护retain_graph

        i = 0
        for layer_name, param in model.named_parameters():
            g1 = grad_l1_list[i]
            g2 = grad_l2_list[i]
            g3 = grad_l3_list[i]
            if g1 is None and g2 is None and g3 is None:
                continue
            elif get_layer_id(layer_name) == '3':  # 模态3的branch里的梯度调节
                if self.mont_loss_3 < self.mont_loss_1 and self.mont_loss_3 < self.mont_loss_2:
                    """模态3是快模态"""
                    c_g3, c_g1, c_g2 = get_fast_modal_grad(g3, g1, g2, version=version)
                    corrected_grad = c_g3 * loss_dict['uc_3'] + c_g1 + c_g2
                else:
                    """模态3是慢模态"""
                    c_g3, c_g1, c_g2 = get_slow_modal_grad(g3, g1, g2, version=version)
                    corrected_grad = c_g3 + c_g1 * loss_dict['uc_1'] + c_g2 * loss_dict['uc_2']
            elif get_layer_id(layer_name) == '2':  # 模态2的branch里的梯度调节
                if self.mont_loss_2 < self.mont_loss_1 and self.mont_loss_2 < self.mont_loss_3:
                    """模态2是快模态"""
                    c_g2, c_g1, c_g3 = get_fast_modal_grad(g2, g1, g3, version=version)
                    corrected_grad = c_g2 * loss_dict['uc_2'] + c_g1 + c_g3
                else:
                    """模态2是慢模态"""
                    c_g2, c_g1, c_g3 = get_slow_modal_grad(g2, g1, g3, version=version)
                    corrected_grad = c_g2 + c_g1 * loss_dict['uc_1'] + c_g3 * loss_dict['uc_3']
            elif get_layer_id(layer_name) == '1':  # 模态1的branch里的梯度调节
                if self.mont_loss_1 < self.mont_loss_2 and self.mont_loss_1 < self.mont_loss_3:
                    """模态1是快模态"""
                    c_g1, c_g2, c_g3 = get_fast_modal_grad(g1, g2, g3, version=version)
                    corrected_grad = c_g1 * loss_dict['uc_1'] + c_g2 + c_g3
                else:
                    """模态1是慢模态"""
                    c_g1, c_g2, c_g3 = get_slow_modal_grad(g1, g2, g3, version=version)
                    corrected_grad = c_g1 + c_g2 * loss_dict['uc_2'] + c_g3 * loss_dict['uc_3']

            if param.grad is not None:
                param.grad += corrected_grad.clone()
            elif param.grad is None and corrected_grad is not None:
                param.grad = corrected_grad.clone()
            """清空梯度，防止显存泄露"""
            g1, g2, g3, c_g1, c_g2, c_g3, corrected_grad = None, None, None, None, None, None, None
            i += 1

        optimizer.step()
        optimizer.zero_grad()
        return temp_loss_dict

    # @torchsnooper.snoop()
    def cal_loss_momentum_missing(self, model, optimizer, loss_dict, version=4, missing=[]):
        loss_1 = loss_dict['m1']
        loss_2 = loss_dict['m2']
        loss_3 = loss_dict['m3']
        loss_total = loss_dict['total']

        # 计算momentum
        if 'RGB' not in missing:
            self.mont_loss_1 += self.mont_loss_1 * self.mont_factor + loss_1.item() * (1.0 - self.mont_factor)
        else:
            self.mont_loss_1 = self.mont_loss_1 * self.mont_factor

        if 'D' not in missing:
            self.mont_loss_2 += self.mont_loss_2 * self.mont_factor + loss_2.item() * (1.0 - self.mont_factor)
        else:
            self.mont_loss_2 = self.mont_loss_2 * self.mont_factor

        if 'IR' not in missing:
            self.mont_loss_3 += self.mont_loss_3 * self.mont_factor + loss_3.item() * (1.0 - self.mont_factor)
        else:
            self.mont_loss_3 = self.mont_loss_3 * self.mont_factor

        # 暂存loss
        temp_loss_dict = {
            "total": torch.tensor(loss_total.item()),
            "m1": torch.tensor(loss_1.item()) if 'RGB' not in missing else 0.0,
            "m2": torch.tensor(loss_2.item()) if 'D' not in missing else 0.0,
            "m3": torch.tensor(loss_3.item()) if 'IR' not in missing else 0.0,
        }

        # 获取各个参数在各个模态的梯度，存放在list中
        # RGB模态梯度列表
        if 'RGB' not in missing:
            optimizer.zero_grad()
            loss_1.backward(retain_graph=True)
            grad_l1_list = get_named_parameters_with_grad(model, 'grad')
        else:
            len_g = len(list(model.named_parameters()))
            grad_l1_list = [None for _ in range(len_g)]
        # depth模态梯度列表
        if 'D' not in missing:
            optimizer.zero_grad()
            loss_2.backward(retain_graph=True)
            grad_l2_list = get_named_parameters_with_grad(model, 'grad')
        else:
            len_g = len(list(model.named_parameters()))
            grad_l2_list = [None for _ in range(len_g)]
        # infrared模态梯度列表
        if 'IR' not in missing:
            optimizer.zero_grad()
            loss_3.backward(retain_graph=True)
            grad_l3_list = get_named_parameters_with_grad(model, 'grad')
        else:
            len_g = len(list(model.named_parameters()))
            grad_l3_list = [None for _ in range(len_g)]

        # 来自分类头的最后一个，不需要维护retain_graph
        optimizer.zero_grad()
        loss_total.backward(retain_graph=False)

        # 遍历每个参数，根据模态间梯度关系，进行梯度调节
        i = 0
        for layer_name, param in model.named_parameters():
            g1 = grad_l1_list[i]
            g2 = grad_l2_list[i]
            g3 = grad_l3_list[i]
            if g1 is None and g2 is None and g3 is None:
                continue
            elif get_layer_id(layer_name) == '3' and 'IR' not in missing:  # 模态3的branch里的梯度调节
                if self.mont_loss_3 < self.mont_loss_1 and self.mont_loss_3 < self.mont_loss_2:
                    """模态3是快模态"""
                    c_g3, c_g1, c_g2 = get_fast_modal_grad(g3, g1, g2, version=version)
                    corrected_grad = c_g3 * loss_dict['uc_3'] + c_g1 + c_g2
                else:
                    """模态3是慢模态"""
                    c_g3, c_g1, c_g2 = get_slow_modal_grad(g3, g1, g2, version=version)
                    corrected_grad = c_g3 + c_g1 * loss_dict['uc_1'] + c_g2 * loss_dict['uc_2']
            elif get_layer_id(layer_name) == '2' and 'D' not in missing:  # 模态2的branch里的梯度调节
                if self.mont_loss_2 < self.mont_loss_1 and self.mont_loss_2 < self.mont_loss_3:
                    """模态2是快模态"""
                    c_g2, c_g1, c_g3 = get_fast_modal_grad(g2, g1, g3, version=version)
                    corrected_grad = c_g2 * loss_dict['uc_2'] + c_g1 + c_g3
                else:
                    """模态2是慢模态"""
                    c_g2, c_g1, c_g3 = get_slow_modal_grad(g2, g1, g3, version=version)
                    corrected_grad = c_g2 + c_g1 * loss_dict['uc_1'] + c_g3 * loss_dict['uc_3']
            elif get_layer_id(layer_name) == '1' and 'RGB' not in missing:  # 模态1的branch里的梯度调节
                if 'D' in missing and 'IR' in missing:
                    corrected_grad = g1
                else:
                    if self.mont_loss_1 < self.mont_loss_2 and self.mont_loss_1 < self.mont_loss_3:
                        """模态1是快模态"""
                        c_g1, c_g2, c_g3 = get_fast_modal_grad(g1, g2, g3, version=version)
                        corrected_grad = c_g1 * loss_dict['uc_1'] + c_g2 + c_g3
                    else:
                        """模态1是慢模态"""
                        c_g1, c_g2, c_g3 = get_slow_modal_grad(g1, g2, g3, version=version)
                        corrected_grad = c_g1 + c_g2 * loss_dict['uc_2'] + c_g3 * loss_dict['uc_3']

            if param.grad is not None and corrected_grad is not None:
                param.grad += corrected_grad.clone()
            elif param.grad is None and corrected_grad is not None:
                param.grad = corrected_grad.clone()
            elif corrected_grad is None:
                pass
            """清空梯度，防止显存泄露"""
            g1, g2, g3, c_g1, c_g2, c_g3, corrected_grad = None, None, None, None, None, None, None
            i += 1

        optimizer.step()
        optimizer.zero_grad()
        return temp_loss_dict
