# neendai web app

next.js web app

## deploy to vercel

### quick deploy

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/aranyoray/neendAI/tree/main/web)

### manual deploy

1. install vercel cli:
```bash
npm i -g vercel
```

2. deploy:
```bash
cd web
vercel
```

3. for production:
```bash
vercel --prod
```

### environment variables

set these in vercel dashboard (settings > environment variables):

- `NEXT_PUBLIC_API_URL` - url of the python backend (optional for demo)
- `JWT_SECRET` - secret for jwt tokens (generate a random string)

## local development

```bash
# install dependencies
npm install

# copy environment template
cp .env.example .env.local

# run dev server
npm run dev
```

open http://localhost:3000

## demo credentials

for testing the deployed app:
- email: demo@neendai.com
- password: demo123

## features

- dark theme (black/navy blue)
- responsive design
- auth flow with validation
- recording interface
- results dashboard
- event timeline
- ahi gauge visualization

## tech stack

- next.js 14 (app router)
- tailwind css
- framer motion
- zustand (state)
- recharts (visualizations)

## project structure

```
web/
├── app/
│   ├── api/          # api routes (serverless)
│   ├── auth/         # signin/signup pages
│   ├── dashboard/    # main dashboard
│   ├── recording/    # audio recording
│   ├── history/      # past recordings
│   ├── settings/     # user settings
│   └── reports/      # detailed reports
├── components/       # react components
├── hooks/           # custom hooks
├── lib/             # utilities
└── public/          # static assets
```

## notes

- api routes work as serverless functions on vercel
- demo mode returns mock analysis results
- for real ml analysis, deploy the python backend separately

## troubleshooting

**build fails:**
- make sure all dependencies are in package.json
- check for typescript errors: `npm run lint`

**api routes not working:**
- check cors headers in next.config.js
- make sure OPTIONS handler is exported

**audio recording fails:**
- needs https (localhost is ok for dev)
- browser needs microphone permission
