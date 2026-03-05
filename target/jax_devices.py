import jax
import jax.numpy as jnp

try:
  devices = jax.devices()
  print("JAX detected the following devices:")
  for i, device in enumerate(devices):
      print(f"{i}: {device.platform.upper()} ({device.device_kind})")

  # Test a simple computation
  key = jax.random.PRNGKey(0)
  x = jax.random.normal(key, (10,))
  y = jnp.dot(x, x)
  print(f"\nSuccessfully executed a simple JAX operation. Result: {y}")

  # Check default device
  print(f"\nDefault device: {jax.default_backend()}")

except Exception as e:
  print("An error occurred during JAX verification:")
  print(e)
  print("\nPlease check your installation steps, especially GPU driver/CUDA versions if applicable.")

