import { View, Text, ScrollView, Pressable, StyleSheet } from 'react-native';
import { router } from 'expo-router';
import Animated, { FadeInDown } from 'react-native-reanimated';

const sessions = [
  { id: '1', date: 'Nov 22', day: 'Friday', duration: '7h 30m', ahi: 12.4, severity: 'Mild', events: 47 },
  { id: '2', date: 'Nov 21', day: 'Thursday', duration: '6h 48m', ahi: 14.1, severity: 'Mild', events: 52 },
  { id: '3', date: 'Nov 20', day: 'Wednesday', duration: '8h 12m', ahi: 8.3, severity: 'Mild', events: 38 },
  { id: '4', date: 'Nov 19', day: 'Tuesday', duration: '7h 06m', ahi: 22.7, severity: 'Moderate', events: 89 },
  { id: '5', date: 'Nov 18', day: 'Monday', duration: '6h 30m', ahi: 18.9, severity: 'Moderate', events: 71 },
];

export default function History() {
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'Normal': return '#4ade80';
      case 'Mild': return '#facc15';
      case 'Moderate': return '#fb923c';
      case 'Severe': return '#f87171';
      default: return '#9ca3af';
    }
  };

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      <View style={styles.header}>
        <Text style={styles.title}>Sleep History</Text>
        <Text style={styles.subtitle}>Your past sleep sessions</Text>
      </View>

      {sessions.map((session, i) => (
        <Animated.View
          key={session.id}
          entering={FadeInDown.delay(i * 100).duration(400)}
        >
          <Pressable style={styles.sessionCard}>
            <View style={styles.sessionLeft}>
              <View style={styles.dateIcon}>
                <Text style={styles.dateIconText}>📅</Text>
              </View>
              <View>
                <Text style={styles.sessionDate}>{session.day}, {session.date}</Text>
                <Text style={styles.sessionMeta}>{session.duration} • {session.events} events</Text>
              </View>
            </View>
            <View style={styles.sessionRight}>
              <Text style={styles.sessionAHI}>{session.ahi.toFixed(1)}</Text>
              <View style={[styles.severityBadge, { backgroundColor: `${getSeverityColor(session.severity)}20` }]}>
                <Text style={[styles.severityText, { color: getSeverityColor(session.severity) }]}>
                  {session.severity}
                </Text>
              </View>
            </View>
          </Pressable>
        </Animated.View>
      ))}

      <Animated.View entering={FadeInDown.delay(600).duration(400)} style={styles.summaryCard}>
        <Text style={styles.summaryTitle}>Weekly Summary</Text>
        <View style={styles.summaryGrid}>
          <View style={styles.summaryItem}>
            <Text style={styles.summaryValue}>15.3</Text>
            <Text style={styles.summaryLabel}>Avg AHI</Text>
          </View>
          <View style={styles.summaryItem}>
            <Text style={styles.summaryValue}>7.2h</Text>
            <Text style={styles.summaryLabel}>Avg Duration</Text>
          </View>
          <View style={styles.summaryItem}>
            <Text style={styles.summaryValue}>297</Text>
            <Text style={styles.summaryLabel}>Total Events</Text>
          </View>
        </View>
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
    marginTop: 60,
    marginBottom: 24,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 14,
    color: '#9ca3af',
  },
  sessionCard: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  sessionLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  dateIcon: {
    width: 40,
    height: 40,
    borderRadius: 8,
    backgroundColor: 'rgba(59,130,246,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  dateIconText: {
    fontSize: 18,
  },
  sessionDate: {
    fontSize: 14,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 2,
  },
  sessionMeta: {
    fontSize: 12,
    color: '#6b7280',
  },
  sessionRight: {
    alignItems: 'flex-end',
  },
  sessionAHI: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 4,
  },
  severityBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 8,
  },
  severityText: {
    fontSize: 10,
    fontWeight: '600',
  },
  summaryCard: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 16,
    padding: 20,
    marginTop: 8,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  summaryTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 16,
  },
  summaryGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  summaryItem: {
    alignItems: 'center',
  },
  summaryValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 4,
  },
  summaryLabel: {
    fontSize: 12,
    color: '#6b7280',
  },
  bottomPadding: {
    height: 40,
  },
});
