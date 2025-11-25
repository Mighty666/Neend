import { useState } from 'react';
import { View, Text, ScrollView, Pressable, StyleSheet, Dimensions } from 'react-native';
import { router } from 'expo-router';
import Animated, { FadeInDown, useAnimatedStyle, useSharedValue, withRepeat, withTiming } from 'react-native-reanimated';
import Svg, { Path, Defs, LinearGradient, Stop } from 'react-native-svg';

const { width } = Dimensions.get('window');

export default function Dashboard() {
  const [currentAHI] = useState(12.4);
  const [isRecording, setIsRecording] = useState(false);

  const pulseScale = useSharedValue(1);

  const pulseStyle = useAnimatedStyle(() => ({
    transform: [{ scale: pulseScale.value }],
  }));

  const startRecording = () => {
    setIsRecording(true);
    pulseScale.value = withRepeat(
      withTiming(1.2, { duration: 1000 }),
      -1,
      true
    );
    router.push('/recording');
  };

  const severity = currentAHI < 5 ? 'Normal' : currentAHI < 15 ? 'Mild' : currentAHI < 30 ? 'Moderate' : 'Severe';
  const severityColor = currentAHI < 5 ? '#4ade80' : currentAHI < 15 ? '#facc15' : currentAHI < 30 ? '#fb923c' : '#f87171';

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      <View style={styles.header}>
        <View>
          <Text style={styles.greeting}>Good Evening</Text>
          <Text style={styles.name}>Maanas</Text>
        </View>
        <View style={styles.avatar}>
          <Text style={styles.avatarText}>M</Text>
        </View>
      </View>

      {/* AHI Card */}
      <Animated.View entering={FadeInDown.delay(100).duration(600)} style={styles.ahiCard}>
        <View style={styles.ahiHeader}>
          <View>
            <Text style={styles.ahiLabel}>Current AHI Score</Text>
            <Text style={styles.ahiValue}>{currentAHI.toFixed(1)}</Text>
          </View>
          <View style={[styles.severityBadge, { backgroundColor: `${severityColor}20` }]}>
            <Text style={[styles.severityText, { color: severityColor }]}>{severity}</Text>
          </View>
        </View>
        <View style={styles.ahiTrend}>
          <Text style={styles.trendDown}>↓ -2.3</Text>
          <Text style={styles.trendLabel}>from last week</Text>
        </View>
      </Animated.View>

      {/* Stats Row */}
      <View style={styles.statsRow}>
        <Animated.View entering={FadeInDown.delay(200).duration(600)} style={styles.statCard}>
          <Text style={styles.statValue}>7h 23m</Text>
          <Text style={styles.statLabel}>Sleep Duration</Text>
        </Animated.View>
        <Animated.View entering={FadeInDown.delay(300).duration(600)} style={styles.statCard}>
          <Text style={styles.statValue}>47</Text>
          <Text style={styles.statLabel}>Total Events</Text>
        </Animated.View>
      </View>

      {/* Mini Chart */}
      <Animated.View entering={FadeInDown.delay(400).duration(600)} style={styles.chartCard}>
        <Text style={styles.chartTitle}>This Week</Text>
        <View style={styles.chartContainer}>
          <Svg width={width - 80} height={100}>
            <Defs>
              <LinearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
                <Stop offset="0%" stopColor="#3b82f6" stopOpacity="0.3" />
                <Stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
              </LinearGradient>
            </Defs>
            <Path
              d={`M 0 60 Q 40 40 80 50 T 160 30 T 240 45 T ${width - 80} 35`}
              stroke="#3b82f6"
              strokeWidth="2"
              fill="none"
            />
          </Svg>
        </View>
        <View style={styles.chartLabels}>
          <Text style={styles.chartLabel}>Mon</Text>
          <Text style={styles.chartLabel}>Tue</Text>
          <Text style={styles.chartLabel}>Wed</Text>
          <Text style={styles.chartLabel}>Thu</Text>
          <Text style={styles.chartLabel}>Fri</Text>
          <Text style={styles.chartLabel}>Sat</Text>
          <Text style={styles.chartLabel}>Sun</Text>
        </View>
      </Animated.View>

      {/* Start Recording */}
      <Animated.View entering={FadeInDown.delay(500).duration(600)} style={styles.recordCard}>
        <Animated.View style={[styles.recordIcon, pulseStyle]}>
        </Animated.View>
        <Text style={styles.recordTitle}>Ready to Sleep?</Text>
        <Text style={styles.recordSubtitle}>
          Start recording to analyze your sleep patterns tonight
        </Text>
        <Pressable style={styles.recordButton} onPress={startRecording}>
          <Text style={styles.recordButtonText}>Start Sleep Analysis</Text>
        </Pressable>
      </Animated.View>

      <View style={styles.bottomPadding} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#09090b',
    padding: 20,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 60,
    marginBottom: 24,
  },
  greeting: {
    fontSize: 14,
    color: '#9ca3af',
  },
  name: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
  },
  avatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#2563eb',
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: {
    color: '#fff',
    fontWeight: '600',
  },
  ahiCard: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  ahiHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  ahiLabel: {
    fontSize: 14,
    color: '#9ca3af',
    marginBottom: 4,
  },
  ahiValue: {
    fontSize: 40,
    fontWeight: 'bold',
    color: '#fff',
  },
  severityBadge: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  severityText: {
    fontSize: 12,
    fontWeight: '600',
  },
  ahiTrend: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  trendDown: {
    color: '#4ade80',
    fontSize: 14,
    fontWeight: '500',
  },
  trendLabel: {
    color: '#6b7280',
    fontSize: 14,
  },
  statsRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  statCard: {
    flex: 1,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  statIcon: {
    fontSize: 20,
    marginBottom: 8,
  },
  statValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 2,
  },
  statLabel: {
    fontSize: 12,
    color: '#6b7280',
  },
  chartCard: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  chartTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 16,
  },
  chartContainer: {
    alignItems: 'center',
  },
  chartLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 8,
  },
  chartLabel: {
    fontSize: 10,
    color: '#6b7280',
  },
  recordCard: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 16,
    padding: 24,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  recordIcon: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: 'rgba(59,130,246,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  moonIcon: {
    fontSize: 28,
  },
  recordTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 8,
  },
  recordSubtitle: {
    fontSize: 14,
    color: '#9ca3af',
    textAlign: 'center',
    marginBottom: 20,
  },
  recordButton: {
    backgroundColor: '#2563eb',
    paddingVertical: 14,
    paddingHorizontal: 32,
    borderRadius: 12,
  },
  recordButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  bottomPadding: {
    height: 40,
  },
});
