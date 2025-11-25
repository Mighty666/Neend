import { useState, useEffect } from 'react';
import { View, Text, Pressable, StyleSheet } from 'react-native';
import { router } from 'expo-router';
import Animated, { useAnimatedStyle, useSharedValue, withRepeat, withTiming, withSequence, FadeIn } from 'react-native-reanimated';

export default function Recording() {
  const [duration, setDuration] = useState(0);
  const [events, setEvents] = useState({ normal: 0, snoring: 0, hypopnea: 0, apnea: 0 });
  const [currentEvent, setCurrentEvent] = useState('normal');

  const pulseScale = useSharedValue(1);
  const waveOpacity = useSharedValue(0.3);

  useEffect(() => {
    // Breathing animation
    pulseScale.value = withRepeat(
      withSequence(
        withTiming(1.15, { duration: 2000 }),
        withTiming(1, { duration: 2000 })
      ),
      -1
    );

    waveOpacity.value = withRepeat(
      withSequence(
        withTiming(0.6, { duration: 2000 }),
        withTiming(0.3, { duration: 2000 })
      ),
      -1
    );

    // Duration timer
    const timer = setInterval(() => {
      setDuration(d => d + 1);
    }, 1000);

    // Simulate events
    const eventTimer = setInterval(() => {
      const rand = Math.random();
      let event = 'normal';
      if (rand > 0.9) event = 'apnea';
      else if (rand > 0.8) event = 'hypopnea';
      else if (rand > 0.6) event = 'snoring';

      setCurrentEvent(event);
      setEvents(e => ({ ...e, [event]: e[event as keyof typeof e] + 1 }));
    }, 3000);

    return () => {
      clearInterval(timer);
      clearInterval(eventTimer);
    };
  }, []);

  const pulseStyle = useAnimatedStyle(() => ({
    transform: [{ scale: pulseScale.value }],
  }));

  const waveStyle = useAnimatedStyle(() => ({
    opacity: waveOpacity.value,
  }));

  const formatDuration = (seconds: number) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  const stopRecording = () => {
    router.replace('/dashboard');
  };

  const eventColors: Record<string, string> = {
    normal: '#4ade80',
    snoring: '#facc15',
    hypopnea: '#fb923c',
    apnea: '#f87171',
  };

  return (
    <View style={styles.container}>
      <Animated.View entering={FadeIn.duration(800)} style={styles.content}>
        {/* Breathing visualization */}
        <View style={styles.visualContainer}>
          <Animated.View style={[styles.waveOuter, waveStyle]} />
          <Animated.View style={[styles.waveMiddle, waveStyle]} />
          <Animated.View style={[styles.pulseCircle, pulseStyle]}>
            <Text style={styles.pulseIcon}>🫁</Text>
          </Animated.View>
        </View>

        <Text style={styles.status}>Recording in Progress</Text>
        <Text style={styles.duration}>{formatDuration(duration)}</Text>

        {/* Current Event */}
        <View style={[styles.currentEventBadge, { backgroundColor: `${eventColors[currentEvent]}20` }]}>
          <View style={[styles.eventDot, { backgroundColor: eventColors[currentEvent] }]} />
          <Text style={[styles.currentEventText, { color: eventColors[currentEvent] }]}>
            {currentEvent.charAt(0).toUpperCase() + currentEvent.slice(1)}
          </Text>
        </View>

        {/* Event Counts */}
        <View style={styles.eventsGrid}>
          {Object.entries(events).map(([key, value]) => (
            <View key={key} style={styles.eventItem}>
              <Text style={styles.eventCount}>{value}</Text>
              <Text style={styles.eventLabel}>{key}</Text>
            </View>
          ))}
        </View>

        {/* AHI Preview */}
        <View style={styles.ahiPreview}>
          <Text style={styles.ahiPreviewLabel}>Current AHI</Text>
          <Text style={styles.ahiPreviewValue}>
            {((events.apnea + events.hypopnea) / Math.max(duration / 3600, 0.01)).toFixed(1)}
          </Text>
        </View>

        {/* Stop Button */}
        <Pressable style={styles.stopButton} onPress={stopRecording}>
          <View style={styles.stopIcon} />
          <Text style={styles.stopText}>Stop Recording</Text>
        </Pressable>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#09090b',
  },
  content: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  visualContainer: {
    width: 200,
    height: 200,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 32,
  },
  waveOuter: {
    position: 'absolute',
    width: 200,
    height: 200,
    borderRadius: 100,
    backgroundColor: 'rgba(59,130,246,0.1)',
  },
  waveMiddle: {
    position: 'absolute',
    width: 150,
    height: 150,
    borderRadius: 75,
    backgroundColor: 'rgba(59,130,246,0.15)',
  },
  pulseCircle: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: 'rgba(59,130,246,0.3)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  pulseIcon: {
    fontSize: 40,
  },
  status: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 8,
  },
  duration: {
    fontSize: 48,
    fontWeight: 'bold',
    color: '#60a5fa',
    fontVariant: ['tabular-nums'],
    marginBottom: 24,
  },
  currentEventBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    marginBottom: 32,
    gap: 8,
  },
  eventDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  currentEventText: {
    fontSize: 14,
    fontWeight: '600',
  },
  eventsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: 16,
    marginBottom: 32,
  },
  eventItem: {
    alignItems: 'center',
    width: 70,
  },
  eventCount: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
  },
  eventLabel: {
    fontSize: 10,
    color: '#6b7280',
    textTransform: 'capitalize',
  },
  ahiPreview: {
    alignItems: 'center',
    marginBottom: 40,
  },
  ahiPreviewLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 4,
  },
  ahiPreviewValue: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#fff',
  },
  stopButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#dc2626',
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderRadius: 12,
    gap: 8,
  },
  stopIcon: {
    width: 12,
    height: 12,
    backgroundColor: '#fff',
    borderRadius: 2,
  },
  stopText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
