import torch

# Verifica si CUDA está disponible
print(f"¿GPU disponible?: {torch.cuda.is_available()}")

# Obtiene el nombre de tu tarjeta
if torch.cuda.is_available():
    print(f"Dispositivo actual: {torch.cuda.get_device_name(0)}")
else:
    print("No se detectó GPU. Revisa los drivers.")