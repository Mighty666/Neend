# neendai mobile app

react native app

## quick start

### prerequisites

- node.js 18+ installed
- ios: xcode 14+ (mac only)
- android: android studio with android sdk

### installation

```bash
# navigate to mobile folder
cd mobile

# install dependencies
npm install

# start the app
npx expo start
```

## running on different platforms

### ios (simulator)

```bash
# press 'i' after npx expo start
# or run directly:
npx expo start --ios
```

**requirements:**
- mac computer
- xcode installed from app store
- ios simulator installed via xcode

### android (emulator)

```bash
# press 'a' after npx expo start
# or run directly:
npx expo start --android
```

**requirements:**
- android studio installed
- android emulator set up (avd manager)
- android sdk platform tools

**setting up android emulator:**
1. open android studio
2. go to tools → device manager
3. create virtual device
4. select pixel 5 or any device
5. download system image (recommended: android 13)
6. start the emulator

### physical device (recommended for audio recording)

**easiest method - expo go app:**

1. install expo go from app store (ios) or play store (android)
2. run `npx expo start`
3. scan qr code with:
   - ios: camera app
   - android: expo go app

**note:** audio recording works better on physical devices than simulators

## troubleshooting

### metro bundler not starting

```bash
# clear cache and restart
npx expo start -c
```

### android build fails

```bash
# make sure android sdk is in path
export ANDROID_HOME=$HOME/Library/Android/sdk
export PATH=$PATH:$ANDROID_HOME/emulator
export PATH=$PATH:$ANDROID_HOME/platform-tools
```

### ios build fails

```bash
# clean build folder
cd ios && xcodebuild clean && cd ..
npx expo start -c
```

### permission errors for audio

the app requests microphone permission automatically. if denied:
- ios: settings → neendai → allow microphone
- android: settings → apps → neendai → permissions → microphone

## connecting to backend

by default, the app connects to the demo api. to use your own backend:

1. edit `lib/api.ts`
2. change `API_BASE_URL` to your backend url
3. make sure backend is running and accessible

## project structure

```
mobile/
├── app/              # screens (expo router)
│   ├── index.tsx     # welcome screen
│   ├── auth/         # login/signup
│   ├── dashboard.tsx # main dashboard
│   ├── recording.tsx # audio recording
│   ├── history.tsx   # past sessions
│   └── settings.tsx  # user settings
├── components/       # reusable components
│   ├── BottomNav.tsx # navigation bar
│   ├── AHIRing.tsx   # circular gauge
│   └── EventBars.tsx # event timeline
├── lib/              # utilities
│   ├── api.ts        # api client
│   ├── store.ts      # state management
│   └── websocket.ts  # real-time streaming
└── package.json
```

## features

- **audio recording** - record overnight sleep audio
- **real-time analysis** - stream audio to backend for processing
- **ahi scoring** - view apnea-hypopnea index results
- **event timeline** - see when events occurred
- **history** - browse past sleep sessions
- **offline mode** - recordings saved locally if no connection

## dev mode shortcuts

when running with `npx expo start`:
- `r` - reload app
- `m` - toggle menu
- `shift+m` - open dev menu on device
- `j` - open debugger

## building for production

### ios (requires apple developer account)

```bash
# install eas cli
npm install -g eas-cli

# login to expo
eas login

# build for ios
eas build --platform ios
```

### android

```bash
# build apk
eas build --platform android --profile preview

# build for play store
eas build --platform android
```

## tips

- use physical device for better audio quality
- keep phone plugged in during recording
- close other apps to prevent interruptions
- test with headphones first to verify recording works

---

started: sept 2023
last updated: jan 2024
