import os
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler


class ADMMStage(nn.Module):
    def __init__(self, d, k):
        super(ADMMStage, self).__init__()
        # Learnable parameters
        self.eta = nn.Parameter(torch.tensor(0.1))
        self.lambda_param = nn.Parameter(torch.tensor(1e-3))
        self.mu_param = nn.Parameter(torch.tensor(1e-3))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

        # Linear layer for X update
        self.linear = nn.Linear(d * k, d * k, bias=False)
        # 初始化linear layer的权重为单位矩阵
        nn.init.eye_(self.linear.weight)

    def x_update(self, X, Y, Z, Lambda, Pi, A):
        d, k = X.size()

        # Compute M: 修正矩阵计算顺序，按照论文中的公式
        M = (self.alpha * (X - Y + Lambda / self.alpha) +
             self.beta * (X - Z + Pi / self.beta) +
             A @ A.t() @ X)

        # Linear approximation: 使用linear layer正确处理矩阵
        X_hat = X - self.eta * (self.linear(M.view(-1)).view(d, k))

        # SVD layer
        U, _, V = torch.svd(X_hat)
        X_new = U @ V.t()

        return X_new

    def y_update(self, X, Lambda):
        # Compute row-wise l2 norm
        row_norms = torch.norm(X + Lambda / self.alpha, dim=1, keepdim=True)
        # Avoid division by zero
        row_norms = torch.clamp(row_norms, min=1e-10)
        # Soft thresholding using ReLU
        scale = torch.relu(row_norms - self.lambda_param / self.alpha) / row_norms
        Y = (X + Lambda / self.alpha) * scale
        return Y

    def z_update(self, X, Pi):
        # Element-wise soft thresholding
        Z = torch.sign(X + Pi / self.beta) * torch.relu(torch.abs(X + Pi / self.beta) - self.mu_param / self.beta)
        return Z

    def forward(self, X, Y, Z, Lambda, Pi, A):
        # Update X
        X_new = self.x_update(X, Y, Z, Lambda, Pi, A)

        # Update Y
        Y_new = self.y_update(X_new, Lambda)

        # Update Z
        Z_new = self.z_update(X_new, Pi)

        # Update Lagrange multipliers
        Lambda_new = Lambda + self.alpha * (X_new - Y_new)
        Pi_new = Pi + self.beta * (X_new - Z_new)

        return X_new, Y_new, Z_new, Lambda_new, Pi_new


class DeepADMM(nn.Module):
    def __init__(self, d, k, num_stages=5):
        super(DeepADMM, self).__init__()
        self.d = d
        self.k = k
        self.num_stages = num_stages

        # Create multiple ADMM stages
        self.stages = nn.ModuleList([ADMMStage(d, k) for _ in range(num_stages)])

        # 添加存储最终状态的变量
        self.final_X = None
        self.final_Y = None
        self.final_Z = None

    def forward(self, A):
        # Initialize variables
        X = torch.randn(self.d, self.k, device=A.device)
        X = self.orthogonalize(X)
        Y = X.clone()
        Z = X.clone()
        Lambda = torch.zeros_like(X)
        Pi = torch.zeros_like(X)

        # Store intermediate results
        X_history = [X]

        # Apply ADMM stages
        for stage in self.stages:
            X, Y, Z, Lambda, Pi = stage(X, Y, Z, Lambda, Pi, A)
            X_history.append(X)

        # 存储最终状态
        self.final_X = X
        self.final_Y = Y
        self.final_Z = Z

        return X, X_history

    @staticmethod
    def orthogonalize(X):
        # Orthogonalize X using SVD
        U, _, V = torch.svd(X)
        return U @ V.t()


def load_data(data_path):
    # X的行是样本，列是特征
    data = sio.loadmat(data_path)
    X = data['X'].astype(np.float32)
    Y = data['Y'].astype(np.float32)
    unique_elements, counts = np.unique(Y, return_counts=True)

    # Normalize the data
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    X_normalized = X_normalized.T

    return torch.FloatTensor(X_normalized), len(unique_elements)


def train_deep_admm(model, A, num_epochs=100, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    A = A.to(device)
    losses = []

    for epoch in range(num_epochs):
        # Forward pass
        X_pred, X_history = model(A)

        # Compute losses correctly
        reconstruction_loss = torch.mean((A - X_pred @ X_pred.t() @ A) ** 2)
        orthogonality_loss = torch.mean((X_pred.t() @ X_pred - torch.eye(model.k, device=device)) ** 2)

        # 使用最终阶段的参数计算正则化损失
        final_stage = model.stages[-1]
        L21_loss = final_stage.lambda_param * torch.sum(torch.norm(X_pred, p=2, dim=1))
        L1_loss = final_stage.mu_param * torch.sum(torch.abs(X_pred))

        loss = reconstruction_loss + orthogonality_loss + L21_loss + L1_loss
        losses.append(loss.item())

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Total Loss: {loss.item():.4f}')
            print(f'Reconstruction: {reconstruction_loss.item():.4f}, Orthogonality: {orthogonality_loss.item():.4f},'
                  f' L21: {L21_loss.item():.4f}, L1: {L1_loss.item():.4f}')

    return model, losses


def export_model_parameters(model):
    """导出模型参数和最终状态"""
    parameters = {
        'stages': [],
        'final_state': {
            'X': model.final_X.detach().cpu().numpy(),
            'Y': model.final_Y.detach().cpu().numpy(),
            'Z': model.final_Z.detach().cpu().numpy()
        }
    }

    # 收集每个阶段的参数
    for i, stage in enumerate(model.stages):
        stage_params = {
            'eta': stage.eta.item(),
            'lambda_param': stage.lambda_param.item(),
            'mu_param': stage.mu_param.item(),
            'alpha': stage.alpha.item(),
            'beta': stage.beta.item(),
            'stage_index': i
        }
        parameters['stages'].append(stage_params)
        # 打印每个阶段的参数
        print(f'Stage {i}: {stage_params}')

    return parameters


def save_results_to_mat(X, Y, Z, filename='results.mat'):
    """将结果保存为.mat文件"""
    sio.savemat(filename, {'X': X, 'Y': Y, 'Z': Z})


if __name__ == "__main__":
    # Set parameters  'cuda' if torch.cuda.is_available() else
    device = torch.device('cpu')

    num_stages = 10

    # 动态读取文件名 USPS_PART umist
    file_path = "COIL20.mat"  # /path/to
    file_name = os.path.basename(file_path)
    file_name = os.path.splitext(file_name)[0]
    print(file_name)  # 输出：USPS_PART.mat

    #  COIL20 USPS_PART    # k 选择的特征数量
    A, k = load_data(file_path)

    d = A.shape[0]  # 特征维度

    # Create model
    model = DeepADMM(d, k, num_stages).to(device)

    # Train model
    model, losses = train_deep_admm(model, A, num_epochs=100, device=device)

    # Test and save results
    with torch.no_grad():
        X_pred, _ = model(A.to(device))

        # Compute test error
        test_error = torch.mean((A.to(device) - X_pred @ X_pred.t() @ A.to(device)) ** 2)
        print(f'Final Test Error: {test_error.item():.4f}')

        # 保存所有结果
        results = {
            'selection_matrix': X_pred.cpu().numpy(),
            'training_losses': np.array(losses),
            'test_error': test_error.item(),
            'model_parameters': export_model_parameters(model)
        }

        param_filename = file_name + '_parameters.mat'
        sio.savemat(param_filename, results)

        # 保存结果到.mat文件
        XYZ_filename = file_name + '_XYZ.mat'
        save_results_to_mat(model.final_X.cpu().numpy(),
                            model.final_Y.cpu().numpy(),
                            model.final_Z.cpu().numpy(),
                            filename=XYZ_filename)

        # 对 Y 进行和下面matlab代码一样的操作
        sqW = (model.final_Y.cpu().numpy() ** 2)
        sumW = np.sum(sqW, axis=1)
        index = np.argsort(sumW, axis=0)
        # 保存index到.mat文件  文件名为file_name + "_index.mat"
        sio.savemat(file_name + "_indices.mat", {'indices': index})

