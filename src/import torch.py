import torch

current_probs = torch.tensor([
    0.0000, 0.1380, 0.0267, 0.0638, 0.0421, 0.0630, 0.0359, 0.0370, 0.0179,
    0.0370, 0.0844, 0.0142, 0.0502, 0.1284, 0.0296, 0.0128, 0.0337, 0.0011,
    0.0533, 0.0531, 0.0155, 0.0019, 0.0265, 0.0067, 0.0056, 0.0157, 0.0059,
    0.0000, 0.0000
])

print("Suma de probabilidades:", current_probs.sum().item())  # Debería ser ~1
print("Valores negativos:", (current_probs < 0).sum().item())  # Debería ser 0

for _ in range(10):  # Ejecuta varias veces para ver los resultados
    next_char_index = torch.multinomial(current_probs, 1).item()
    print("Índice seleccionado:", next_char_index)
