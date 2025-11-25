import { View, Text, StyleSheet } from 'react-native';
import Svg, { Circle } from 'react-native-svg';
import Animated, { useAnimatedProps, withTiming, useSharedValue, withDelay } from 'react-native-reanimated';

const AnimatedCircle = Animated.createAnimatedComponent(Circle);

interface AHIRingProps {
  value: number;
  maxValue?: number;
  size?: number;
}

export function AHIRing({ value, maxValue = 60, size = 120 }: AHIRingProps) {
  const percentage = Math.min((value / maxValue) * 100, 100);
  const circumference = 2 * Math.PI * 45;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  const progress = useSharedValue(circumference);

  // Animate on mount
  React.useEffect(() => {
    progress.value = withDelay(300, withTiming(strokeDashoffset, { duration: 1000 }));
  }, [strokeDashoffset]);

  const animatedProps = useAnimatedProps(() => ({
    strokeDashoffset: progress.value,
  }));

  const getColor = () => {
    if (value < 5) return '#4ade80';
    if (value < 15) return '#facc15';
    if (value < 30) return '#fb923c';
    return '#f87171';
  };

  const getSeverity = () => {
    if (value < 5) return 'Normal';
    if (value < 15) return 'Mild';
    if (value < 30) return 'Moderate';
    return 'Severe';
  };

  return (
    <View style={[styles.container, { width: size, height: size }]}>
      <Svg viewBox="0 0 100 100" style={{ width: size, height: size, transform: [{ rotate: '-90deg' }] }}>
        <Circle
          cx="50"
          cy="50"
          r="45"
          fill="none"
          stroke="rgba(255,255,255,0.1)"
          strokeWidth="8"
        />
        <AnimatedCircle
          cx="50"
          cy="50"
          r="45"
          fill="none"
          stroke={getColor()}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          animatedProps={animatedProps}
        />
      </Svg>
      <View style={styles.center}>
        <Text style={[styles.value, { color: getColor() }]}>{value.toFixed(1)}</Text>
        <Text style={styles.label}>AHI</Text>
        <Text style={[styles.severity, { color: getColor() }]}>{getSeverity()}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'relative',
  },
  center: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
  },
  value: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  label: {
    fontSize: 10,
    color: '#9ca3af',
  },
  severity: {
    fontSize: 10,
    marginTop: 2,
  },
});

import React from 'react';
