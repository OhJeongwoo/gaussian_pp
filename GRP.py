import torch


class GRP:
    def __init__(self, A, N, T, eps, kernel_type):
        self.A = A
        self.N = N
        self.T = T
        self.eps = eps
        self.kernel_type = kernel_type


    def generate_random_path(self):
        return 0
    
    def solve(self):
