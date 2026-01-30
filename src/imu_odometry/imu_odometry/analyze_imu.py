#!/usr/bin/env python3
# analyze_pure_imu.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('pure_imu_odometry.csv')

# Calculate time from start
df['t'] = df['time'] - df['time'].iloc[0]

# Calculate statistics
print("="*60)
print("PURE IMU ODOMETRY ANALYSIS")
print("="*60)

# 1. Acceleration statistics (should be ~0 when stationary)
stationary_mask = df['t'] < 30  # First 30 seconds
accel_std = df.loc[stationary_mask, ['ax', 'ay', 'az']].std()
print("\n1. ACCELERATION NOISE (stationary):")
print(f"   X std: {accel_std['ax']*100:.1f} cm/s²")
print(f"   Y std: {accel_std['ay']*100:.1f} cm/s²")
print(f"   Z std: {accel_std['az']*100:.1f} cm/s²")

# 2. Velocity drift rate
vx_start = df.loc[stationary_mask, 'vx'].iloc[0]
vx_end = df.loc[stationary_mask, 'vx'].iloc[-1]
time_span = df.loc[stationary_mask, 't'].iloc[-1] - df.loc[stationary_mask, 't'].iloc[0]
velocity_drift_rate = (vx_end - vx_start) / time_span if time_span > 0 else 0

print(f"\n2. VELOCITY DRIFT RATE (stationary):")
print(f"   Drift rate: {velocity_drift_rate*100:.3f} cm/s²")
print(f"   Equivalent bias: {velocity_drift_rate:.5f} m/s²")

# 3. Position drift
px_start = df.loc[stationary_mask, 'px'].iloc[0]
px_end = df.loc[stationary_mask, 'px'].iloc[-1]
position_drift = px_end - px_start

print(f"\n3. POSITION DRIFT (30s stationary):")
print(f"   Drift: {position_drift*100:.1f} cm")
print(f"   Drift rate: {position_drift/30*100:.1f} cm/s")

# 4. Impulse response (if any)
if len(df) > stationary_mask.sum():
    impulse_start = df.loc[~stationary_mask, 't'].iloc[0]
    impulse_duration = 5  # Look at 5s after impulse
    
    impulse_mask = (df['t'] > impulse_start) & (df['t'] < impulse_start + impulse_duration)
    
    if impulse_mask.any():
        vx_before = df.loc[stationary_mask, 'vx'].iloc[-1]
        vx_after = df.loc[impulse_mask, 'vx'].iloc[-1]
        velocity_change = vx_after - vx_before
        
        print(f"\n4. IMPULSE RESPONSE:")
        print(f"   Velocity change: {velocity_change:.3f} m/s")
        
        # Check if velocity decays (should NOT without damping)
        vx_max = df.loc[impulse_mask, 'vx'].max()
        vx_final = df.loc[impulse_mask, 'vx'].iloc[-1]
        
        if abs(vx_final - vx_max) > 0.01:
            print(f"   Velocity decay: {vx_final - vx_max:.3f} m/s (unexpected!)")

print("\n" + "="*60)
print("FUNDAMENTAL LIMITS:")
print("="*60)

# Calculate theoretical limits
accel_noise = accel_std.mean()
theoretical_position_error = 0.5 * accel_noise * 30**2  # t² growth

print(f"Acceleration noise level: {accel_noise*100:.2f} cm/s²")
print(f"Theoretical 30s position error: {theoretical_position_error*100:.1f} cm")

# Recommendations based on measurements
print("\n" + "="*60)
print("RECOMMENDATIONS:")
print("="*60)

if position_drift < 0.1:  # <10cm drift in 30s
    print("✓ EXCEPTIONAL: IMU has very low bias")
    print("  Can use with simple ZUPT for short-term navigation")
elif position_drift < 0.5:  # <50cm drift in 30s
    print("✓ GOOD: IMU has reasonable bias")
    print("  Need bias estimation + ZUPT")
    print("  Wheel encoder fusion recommended for >10s navigation")
elif position_drift < 2.0:  # <2m drift in 30s
    print("⚠ MODERATE: IMU has significant bias")
    print("  Must use: Bias estimation + ZUPT + Wheel encoders")
    print("  GNSS recommended for >30s operation")
else:
    print("⚠ POOR: IMU has large bias/noise")
    print("  Requires: Kalman filter + Wheel encoders + GNSS")
    print("  Not suitable for IMU-only navigation")

# Plot results
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# Acceleration
axes[0,0].plot(df['t'], df['ax'], 'r', alpha=0.7, label='X')
axes[0,0].plot(df['t'], df['ay'], 'g', alpha=0.7, label='Y')
axes[0,0].plot(df['t'], df['az'], 'b', alpha=0.7, label='Z')
axes[0,0].axvline(x=30, color='k', linestyle='--', label='Impulse start')
axes[0,0].set_ylabel('Acceleration (m/s²)')
axes[0,0].set_title('Raw Acceleration')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Velocity
axes[1,0].plot(df['t'], df['vx'], 'r', label='X')
axes[1,0].plot(df['t'], df['vy'], 'g', label='Y')
axes[1,0].plot(df['t'], df['vz'], 'b', label='Z')
axes[1,0].axvline(x=30, color='k', linestyle='--')
axes[1,0].set_ylabel('Velocity (m/s)')
axes[1,0].set_title('Integrated Velocity')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Position
axes[2,0].plot(df['t'], df['px'], 'r', label='X')
axes[2,0].plot(df['t'], df['py'], 'g', label='Y')
axes[2,0].plot(df['t'], df['pz'], 'b', label='Z')
axes[2,0].axvline(x=30, color='k', linestyle='--')
axes[2,0].set_xlabel('Time (s)')
axes[2,0].set_ylabel('Position (m)')
axes[2,0].set_title('Double-Integrated Position')
axes[2,0].legend()
axes[2,0].grid(True, alpha=0.3)

# Histogram of acceleration (stationary)
axes[0,1].hist(df.loc[stationary_mask, 'ax'], bins=50, alpha=0.7, color='r', density=True, label='X')
axes[0,1].hist(df.loc[stationary_mask, 'ay'], bins=50, alpha=0.7, color='g', density=True, label='Y')
axes[0,1].hist(df.loc[stationary_mask, 'az'], bins=50, alpha=0.7, color='b', density=True, label='Z')
axes[0,1].set_xlabel('Acceleration (m/s²)')
axes[0,1].set_ylabel('Density')
axes[0,1].set_title('Acceleration Distribution (Stationary)')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Position drift over time
axes[1,1].plot(df.loc[stationary_mask, 't'], df.loc[stationary_mask, 'px']*100, 'r', label='X')
axes[1,1].plot(df.loc[stationary_mask, 't'], df.loc[stationary_mask, 'py']*100, 'g', label='Y')
axes[1,1].plot(df.loc[stationary_mask, 't'], df.loc[stationary_mask, 'pz']*100, 'b', label='Z')
axes[1,1].set_xlabel('Time (s)')
axes[1,1].set_ylabel('Position Drift (cm)')
axes[1,1].set_title('Position Drift (First 30s)')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

# Remove empty subplot
axes[2,1].axis('off')

plt.tight_layout()
plt.savefig('pure_imu_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to pure_imu_analysis.png")

plt.show()