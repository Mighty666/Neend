import { useState } from 'react';
import { View, Text, ScrollView, Pressable, StyleSheet, Switch, TextInput } from 'react-native';
import Animated, { FadeInDown } from 'react-native-reanimated';

export default function Settings() {
  const [alertThreshold, setAlertThreshold] = useState(30);
  const [notifications, setNotifications] = useState({
    criticalAlerts: true,
    dailySummary: true,
    weeklyReport: false,
  });
  const [privacy, setPrivacy] = useState({
    shareWithClinician: false,
    contributeToResearch: false,
  });

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      <View style={styles.header}>
        <Text style={styles.title}>Settings</Text>
        <Text style={styles.subtitle}>Manage your preferences</Text>
      </View>

      {/* Profile */}
      <Animated.View entering={FadeInDown.delay(100).duration(400)} style={styles.section}>
        <Text style={styles.sectionTitle}>👤 Profile</Text>
        <View style={styles.inputGroup}>
          <Text style={styles.label}>Name</Text>
          <TextInput
            style={styles.input}
            defaultValue="Maanas"
            placeholderTextColor="#6b7280"
          />
        </View>
        <View style={styles.inputGroup}>
          <Text style={styles.label}>Email</Text>
          <TextInput
            style={styles.input}
            defaultValue="maanas@example.com"
            placeholderTextColor="#6b7280"
            keyboardType="email-address"
          />
        </View>
        <View style={styles.inputGroup}>
          <Text style={styles.label}>Emergency Contact</Text>
          <TextInput
            style={styles.input}
            placeholder="+1 (555) 000-0000"
            placeholderTextColor="#6b7280"
            keyboardType="phone-pad"
          />
        </View>
      </Animated.View>

      {/* Notifications */}
      <Animated.View entering={FadeInDown.delay(200).duration(400)} style={styles.section}>
        <Text style={styles.sectionTitle}>🔔 Notifications</Text>

        <View style={styles.sliderGroup}>
          <Text style={styles.label}>Alert Threshold (AHI): {alertThreshold}</Text>
          <View style={styles.sliderContainer}>
            <Text style={styles.sliderLabel}>5</Text>
            <View style={styles.slider}>
              {/* Simplified slider - use @react-native-community/slider in production */}
              <View style={[styles.sliderFill, { width: `${((alertThreshold - 5) / 55) * 100}%` }]} />
            </View>
            <Text style={styles.sliderLabel}>60</Text>
          </View>
        </View>

        {Object.entries(notifications).map(([key, value]) => (
          <View key={key} style={styles.switchRow}>
            <Text style={styles.switchLabel}>
              {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
            </Text>
            <Switch
              value={value}
              onValueChange={(v) => setNotifications({ ...notifications, [key]: v })}
              trackColor={{ false: '#1a1a1f', true: '#2563eb' }}
              thumbColor="#fff"
            />
          </View>
        ))}
      </Animated.View>

      {/* Privacy */}
      <Animated.View entering={FadeInDown.delay(300).duration(400)} style={styles.section}>
        <Text style={styles.sectionTitle}>🔒 Privacy</Text>

        <View style={styles.switchRow}>
          <View>
            <Text style={styles.switchLabel}>Share with Clinician</Text>
            <Text style={styles.switchDescription}>Allow your doctor to view reports</Text>
          </View>
          <Switch
            value={privacy.shareWithClinician}
            onValueChange={(v) => setPrivacy({ ...privacy, shareWithClinician: v })}
            trackColor={{ false: '#1a1a1f', true: '#2563eb' }}
            thumbColor="#fff"
          />
        </View>

        <View style={styles.switchRow}>
          <View>
            <Text style={styles.switchLabel}>Contribute to Research</Text>
            <Text style={styles.switchDescription}>Anonymized data improves accuracy</Text>
          </View>
          <Switch
            value={privacy.contributeToResearch}
            onValueChange={(v) => setPrivacy({ ...privacy, contributeToResearch: v })}
            trackColor={{ false: '#1a1a1f', true: '#2563eb' }}
            thumbColor="#fff"
          />
        </View>
      </Animated.View>

      {/* Actions */}
      <Animated.View entering={FadeInDown.delay(400).duration(400)} style={styles.actions}>
        <Pressable style={styles.saveButton}>
          <Text style={styles.saveButtonText}>Save Changes</Text>
        </Pressable>
        <Pressable style={styles.signOutButton}>
          <Text style={styles.signOutButtonText}>Sign Out</Text>
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
  section: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 16,
  },
  inputGroup: {
    marginBottom: 12,
  },
  label: {
    fontSize: 12,
    color: '#9ca3af',
    marginBottom: 6,
  },
  input: {
    backgroundColor: '#1a1a1f',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
    borderRadius: 8,
    padding: 12,
    color: '#fff',
    fontSize: 14,
  },
  sliderGroup: {
    marginBottom: 16,
  },
  sliderContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginTop: 8,
  },
  slider: {
    flex: 1,
    height: 4,
    backgroundColor: '#1a1a1f',
    borderRadius: 2,
  },
  sliderFill: {
    height: '100%',
    backgroundColor: '#2563eb',
    borderRadius: 2,
  },
  sliderLabel: {
    fontSize: 10,
    color: '#6b7280',
  },
  switchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
  },
  switchLabel: {
    fontSize: 14,
    color: '#fff',
  },
  switchDescription: {
    fontSize: 10,
    color: '#6b7280',
    marginTop: 2,
  },
  actions: {
    gap: 12,
  },
  saveButton: {
    backgroundColor: '#2563eb',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  saveButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  signOutButton: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  signOutButtonText: {
    color: '#f87171',
    fontSize: 16,
    fontWeight: '600',
  },
  bottomPadding: {
    height: 40,
  },
});
