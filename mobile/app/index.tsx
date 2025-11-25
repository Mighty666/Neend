import { View, Text, Pressable, StyleSheet } from 'react-native';
import { Link } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, { FadeIn, FadeInDown } from 'react-native-reanimated';

export default function Home() {
  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#09090b', '#0f172a']}
        style={StyleSheet.absoluteFill}
      />

      <Animated.View entering={FadeIn.duration(800)} style={styles.logoContainer}>
        <Text style={styles.logo}>
          <Text style={styles.logoGradient}>Neend</Text>
          <Text style={styles.logoWhite}>AI</Text>
        </Text>
      </Animated.View>

      <Animated.View entering={FadeInDown.delay(200).duration(800)} style={styles.content}>
        <Text style={styles.title}>
          Know Your Breath.{'\n'}
          <Text style={styles.titleGradient}>Own Your Night.</Text>
        </Text>

        <Text style={styles.subtitle}>
          AI-powered sleep apnea detection. Clinical-grade analysis. Zero lab visits.
        </Text>

        <View style={styles.stats}>
          {[
            { value: '95%', label: 'Accuracy' },
            { value: '<2s', label: 'Detection' },
            { value: '24/7', label: 'Monitoring' },
          ].map((stat, i) => (
            <View key={i} style={styles.statItem}>
              <Text style={styles.statValue}>{stat.value}</Text>
              <Text style={styles.statLabel}>{stat.label}</Text>
            </View>
          ))}
        </View>
      </Animated.View>

      <Animated.View entering={FadeInDown.delay(400).duration(800)} style={styles.buttons}>
        <Link href="/auth/signup" asChild>
          <Pressable style={styles.primaryButton}>
            <Text style={styles.primaryButtonText}>Get Started</Text>
          </Pressable>
        </Link>

        <Link href="/auth/signin" asChild>
          <Pressable style={styles.secondaryButton}>
            <Text style={styles.secondaryButtonText}>Sign In</Text>
          </Pressable>
        </Link>
      </Animated.View>

      <Text style={styles.footer}>
        Sleep better. Live better.
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 24,
    justifyContent: 'space-between',
  },
  logoContainer: {
    marginTop: 60,
    alignItems: 'center',
  },
  logo: {
    fontSize: 32,
    fontWeight: 'bold',
  },
  logoGradient: {
    color: '#60a5fa',
  },
  logoWhite: {
    color: '#fff',
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  title: {
    fontSize: 36,
    fontWeight: 'bold',
    color: '#fff',
    textAlign: 'center',
    marginBottom: 16,
    lineHeight: 44,
  },
  titleGradient: {
    color: '#60a5fa',
  },
  subtitle: {
    fontSize: 16,
    color: '#9ca3af',
    textAlign: 'center',
    marginBottom: 32,
    paddingHorizontal: 20,
  },
  stats: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 32,
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#60a5fa',
  },
  statLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 4,
  },
  buttons: {
    gap: 12,
    marginBottom: 20,
  },
  primaryButton: {
    backgroundColor: '#2563eb',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  primaryButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  secondaryButton: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  secondaryButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  footer: {
    textAlign: 'center',
    color: '#4b5563',
    fontSize: 12,
  },
});
