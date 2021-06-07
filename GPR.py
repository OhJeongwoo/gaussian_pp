import torch
import numpy as np  

class GPR():
    def __init__(self):
        self.l = torch.tensor(1e1, requires_grad=True)
        self.sigma_f = torch.tensor(1e0, requires_grad=True)
        self.sigma_y = torch.tensor(1e-4, requires_grad=True)
        self.reset()
        self.prev_loss = 0
        self.max_step = 10
        self.threshold = 1e-3
        

        

    def load_data(self, X_train_tensor, Y_train_tensor):
        self.X_train_tensor = X_train_tensor
        self.Y_train_tensor = Y_train_tensor
        self.Y_train_mean = Y_train_tensor.mean()
        self.Y_train_tensor = self.Y_train_tensor - self.Y_train_mean
        
        

    def reset(self):
        self.parameters = [self.l, self.sigma_f, self.sigma_y]
        self.optimizer = torch.optim.Adam(self.parameters, lr=1e-3)
        


    def GaussianRBF(self, X, Y):
        assert X.shape[1] == Y.shape[1] # dimension must be equal!
        m, d = X.shape
        n, d = Y.shape
        term1 = torch.sum(X**2, dim=1).view(-1, 1)
        assert term1.shape == (m, 1)
        term2 = torch.matmul(X, Y.t())
        assert term2.shape == (m, n)
        term3 = torch.sum(Y**2, dim=1).view(1, -1)
        assert term3.shape == (1, n)
        dist_matrix = (term1 - 2*term2 + term3)
        return (self.sigma_f**2) * torch.exp((-1/(self.l**2)) * dist_matrix)

    def predict_posterior(self, X, X_train, Y_train):
        K11 = self.GaussianRBF(X_train, X_train) + self.sigma_y**2 * torch.eye(len(X_train))
        K21 = self.GaussianRBF(X, X_train)
        K12 = self.GaussianRBF(X_train, X) # Note, K12 = K21.T (transposed)
        K22 = self.GaussianRBF(X, X)
        K11_inv = torch.linalg.inv(K11)
        mu = torch.matmul(K21, torch.matmul(K11_inv, Y_train)) + self.Y_train_mean
        cov = K22 - torch.matmul(K21, torch.matmul(K11_inv, K12))
        return mu, cov

    def set_hyperparameter(self, l, sigma_f, sigma_y):
        self.l = torch.tensor(l, requires_grad=True)
        self.sigma_f = torch.tensor(sigma_f, requires_grad=True)
        self.sigma_y = torch.tensor(sigma_y, requires_grad=True)
        self.reset()

    def get_hyperparameter(self):
        return self.sigma_f.item(), self.l.item(), self.sigma_y.item()

    def optimize(self):
        for i in range(self.max_step):
            K = self.GaussianRBF(self.X_train_tensor, self.X_train_tensor)
            K = K + ((self.sigma_y+1e-10)**2) * torch.eye(len(self.X_train_tensor))
            loss = 0.5*torch.matmul(self.Y_train_tensor.t(), torch.matmul(K.inverse(), self.Y_train_tensor)) + 0.5*torch.log(torch.det(K))
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            if abs(loss.item() - self.prev_loss) < self.threshold:
                break
            self.prev_loss = loss.item()
            # if True:
            #     print('Iter[{}/5000]\tloss:{:.4f}\tl:{:.4f}\tsigma_f:{:.4f}\tsigma_y:{:.4f}'.format(i+1, loss.item(), self.l.item(), self.sigma_f.item(), self.sigma_y.item()))

        