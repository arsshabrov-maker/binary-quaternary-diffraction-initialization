import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

def my_custom_init(tensor):
    n_in = tensor.size(1)
    n_out = tensor.size(0)
    qwerty = n_in -(math.isqrt(n_in)**2)
    if qwerty==0 :
        std = 0.001*n_out
    elif qwerty !=0 :
        if  (n_in -(math.isqrt(n_in)**2))/2==(n_in -(math.isqrt(n_in)**2))//2 :
            if (n_in -(math.isqrt(n_in)**2))/4==(n_in -(math.isqrt(n_in)**2))//4:
                std=0.00025*n_out
            else:
                std =0.0005*n_out



    with torch.no_grad():
        return tensor.normal_(0, std)

def xavier_manual_init(tensor):
    n_in = tensor.size(1)
    n_out = tensor.size(0)
    std = math.sqrt(2.0 / (n_in + n_out))
    with torch.no_grad():
        return tensor.normal_(0, std)

class DeepProbeNet(nn.Module):
    def __init__(self, mode="xavier"):
        super().__init__()
        layers = []
        for _ in range(20): # ОЧЕНЬ глубоко
            layers.append(nn.Linear(100, 100))
            layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

        # Применяем инициализацию
        for m in self.model:
            if isinstance(m, nn.Linear):
                if mode == "xavier": xavier_manual_init(m.weight)
                else: my_custom_init(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        activations = []
        for layer in self.model:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                activations.append(x.std().item()) # Запоминаем разброс сигнала
        return activations

# --- ТЕСТ ---
x = torch.randn(1, 100) # Входной сигнал с дисперсией 1.0

std_xavier = DeepProbeNet(mode="xavier")(x)
std_custom = DeepProbeNet(mode="custom")(x)

# Визуализация "Живучести" сигнала
plt.figure(figsize=(10, 5))
plt.plot(std_xavier, label='Xavier Math (Standard)', marker='o')
plt.plot(std_custom, label='Your Custom Logic', marker='s', lw=2)
plt.axhline(y=1.0, color='r', linestyle='--', label='Ideal Stability')
plt.title('Как быстро умирает сигнал в 20 слоях')
plt.xlabel('Номер слоя')
plt.ylabel('Дисперсия (Сила сигнала)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Сила сигнала на выходе Xavier: {std_xavier[-1]:.6f}")
print(f"Сила сигнала на выходе Custom: {std_custom[-1]:.6f}")
