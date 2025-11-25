import { View, Text, StyleSheet } from 'react-native';
import Animated, { FadeIn } from 'react-native-reanimated';

interface Event {
  type: 'normal' | 'snoring' | 'hypopnea' | 'apnea';
}

interface EventBarsProps {
  events: Event[];
  maxEvents?: number;
}

export function EventBars({ events, maxEvents = 30 }: EventBarsProps) {
  const displayEvents = events.slice(-maxEvents);

  const eventColors = {
    normal: '#4ade80',
    snoring: '#facc15',
    hypopnea: '#fb923c',
    apnea: '#f87171',
  };

  const eventHeights = {
    normal: 8,
    snoring: 12,
    hypopnea: 16,
    apnea: 20,
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Recent Events</Text>
      <View style={styles.barsContainer}>
        {displayEvents.map((event, i) => (
          <Animated.View
            key={i}
            entering={FadeIn.delay(i * 20).duration(200)}
            style={[
              styles.bar,
              {
                backgroundColor: eventColors[event.type],
                height: eventHeights[event.type],
              },
            ]}
          />
        ))}
      </View>
      <View style={styles.legend}>
        {Object.entries(eventColors).map(([name, color]) => (
          <View key={name} style={styles.legendItem}>
            <View style={[styles.legendDot, { backgroundColor: color }]} />
            <Text style={styles.legendText}>{name}</Text>
          </View>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  title: {
    fontSize: 14,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 12,
  },
  barsContainer: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    height: 24,
    gap: 2,
  },
  bar: {
    flex: 1,
    borderTopLeftRadius: 2,
    borderTopRightRadius: 2,
    opacity: 0.8,
  },
  legend: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 12,
    marginTop: 12,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  legendDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  legendText: {
    fontSize: 8,
    color: '#6b7280',
    textTransform: 'capitalize',
  },
});
