import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';

export default function RootLayout() {
  return (
    <>
      <StatusBar style="light" />
      <Stack
        screenOptions={{
          headerStyle: {
            backgroundColor: '#09090b',
          },
          headerTintColor: '#fff',
          headerTitleStyle: {
            fontWeight: 'bold',
          },
          contentStyle: {
            backgroundColor: '#09090b',
          },
        }}
      >
        <Stack.Screen name="index" options={{ headerShown: false }} />
        <Stack.Screen name="auth/signin" options={{ title: 'Sign In', headerShown: false }} />
        <Stack.Screen name="auth/signup" options={{ title: 'Sign Up', headerShown: false }} />
        <Stack.Screen name="dashboard" options={{ title: 'Dashboard', headerShown: false }} />
        <Stack.Screen name="recording" options={{ title: 'Recording', headerShown: false }} />
      </Stack>
    </>
  );
}
