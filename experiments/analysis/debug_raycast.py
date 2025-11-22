"""
Debug raycasting to see what's happening
"""
import numpy as np
import math

# Simulate the raycasting
num_rays = 16
ray_length = 15

# Calculate ray directions (same as in expanded observer)
angles = [i * (2 * math.pi / num_rays) for i in range(num_rays)]
ray_directions = [
    (math.cos(angle), math.sin(angle))
    for angle in angles
]

print("Ray directions:")
for i, (dx, dy) in enumerate(ray_directions):
    print(f"  Ray {i}: ({dx:.3f}, {dy:.3f})")

# Test raycasting DOWN
print("\n\nTest: Ray pointing DOWN (should be ray 4 with 16 rays)")
print("Snake head at (10, 10), body at (10, 11), (10, 12), (10, 13)")

# Ray 4 should point roughly down
ray_idx = 4
dx, dy = ray_directions[ray_idx]
print(f"\nRay {ray_idx}: direction ({dx:.3f}, {dy:.3f})")

ax, ay = 10, 10
snake_body = {(10, 11), (10, 12), (10, 13)}

print("\nRaycasting:")
for step in range(1, 16):
    x = ax + dx * step
    y = ay + dy * step
    px, py = int(round(x)), int(round(y))

    if (px, py) in snake_body:
        print(f"  Step {step}: ({x:.2f}, {y:.2f}) -> ({px}, {py}) [HIT BODY!]")
        break
    else:
        print(f"  Step {step}: ({x:.2f}, {y:.2f}) -> ({px}, {py})")

# Try a few more rays around that direction
print("\n\nTrying adjacent rays:")
for test_idx in [3, 4, 5, 8, 9]:
    dx, dy = ray_directions[test_idx]
    print(f"\nRay {test_idx}: direction ({dx:.3f}, {dy:.3f})")

    found = False
    for step in range(1, 5):
        x = ax + dx * step
        y = ay + dy * step
        px, py = int(round(x)), int(round(y))

        if (px, py) in snake_body:
            print(f"  Step {step}: FOUND body at ({px}, {py})")
            found = True
            break

    if not found:
        print(f"  No body found in first 5 steps")
