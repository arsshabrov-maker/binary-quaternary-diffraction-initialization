import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt

# ТВОЯ УЛУЧШЕННАЯ ЛОГИКА (SRLI)
def srli_init(tensor):
    n_in = tensor.size(1)
    n_out = tensor.size(1)
    depth_factor = math.sqrt(100)
    with torch.no_grad():
        qwerty = n_in*n_out - (math.isqrt(n_in)**2)*(math.isqrt(n_out)**2)
        if qwerty == 0:
            std = 0.001*n_out/depth_factor
        else:
            # Твоя проверка на кратность 2 и 4
            if qwerty % 2 == 0:
                if qwerty % 4 == 0:
                    std = 0.0025*n_out/depth_factor
                else:
                    std = 0.005*n_out/depth_factor
            else:
                std = 0.001*n_out/depth_factor

        new_weights = torch.normal(0, std, size=tensor.size())
        tensor.copy_(new_weights)

def xavier_init(tensor):
    nn.init.xavier_normal_(tensor)

class MegaDeepNet(nn.Module):
    def __init__(self, mode="srli"):
        super().__init__()
        layers = []
        # 100 СЛОЕВ — это зона смерти для обычных сетей
        width = 104
        for _ in range(100):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, 1))
        self.model = nn.Sequential(*layers)

        for m in self.model:
            if isinstance(m, nn.Linear):
                if mode == "srli": srli_init(m.weight)
                else: xavier_init(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x): return self.model(x)

# Данные для обучения
x_train = torch.randn(64, 104)
y_target = torch.randn(64, 1)

def train_mega(mode):
    model = MegaDeepNet(mode=mode)
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Маленький шаг для 100 слоев
    criterion = nn.MSELoss()
    losses = []

    for i in range(2000): # Даем 2000 итераций, чтобы пробить 100 слоев
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if i % 500 == 0: print(f"{mode} step {i}... Loss: {loss.item():.6f}")
    return losses

# ЗАПУСК ТИТАНОВ
print("Запуск теста '100 слоев'...")
history_srli = train_mega("srli")
history_xavier = train_mega("xavier")

# ГРАФИК РЕКОРДА
plt.figure(figsize=(12, 7))
plt.plot(history_xavier, label='Xavier (Classic)', color='blue', alpha=0.5)
plt.plot(history_srli, label='SRLI (Arseniy\'s Logic)', color='red', lw=2)
plt.yscale('log')
plt.title('100-Layer Survival Test: SRLI vs Xavier')
plt.xlabel('Iteration')
plt.ylabel('Loss (log scale)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.show()

print(f"Финальный результат (Xavier): {history_xavier[-1]:.10f}")
print(f"Финальный результат (SRLI):   {history_srli[-1]:.10f}")
