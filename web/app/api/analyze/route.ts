import { NextResponse } from 'next/server'

// mock analysis for demo since we dont have the python backend on vercel
// in production this would call the actual ml api

interface AnalysisResult {
  sleepQuality: number
  ahiScore: number
  severity: string
  events: Array<{
    type: string
    timestamp: number
    duration: number
    confidence: number
  }>
  recommendations: string[]
}

function generateMockAnalysis(): AnalysisResult {
  // generate realistic-ish random results for demo
  const ahiScore = Math.random() * 30
  let severity = 'normal'
  if (ahiScore >= 5) severity = 'mild'
  if (ahiScore >= 15) severity = 'moderate'
  if (ahiScore >= 30) severity = 'severe'

  const sleepQuality = Math.max(20, 100 - ahiScore * 2 + (Math.random() - 0.5) * 20)

  // generate some random events
  const numEvents = Math.floor(ahiScore * 8 / 30)
  const events = []
  for (let i = 0; i < numEvents; i++) {
    events.push({
      type: Math.random() > 0.3 ? 'apnea' : 'hypopnea',
      timestamp: Math.floor(Math.random() * 28800),
      duration: 10 + Math.random() * 30,
      confidence: 0.7 + Math.random() * 0.3
    })
  }

  events.sort((a, b) => a.timestamp - b.timestamp)

  // recommendations based on severity
  const recommendations = []
  if (severity === 'normal') {
    recommendations.push('your sleep breathing appears normal')
    recommendations.push('maintain good sleep hygiene')
  } else if (severity === 'mild') {
    recommendations.push('mild sleep apnea detected')
    recommendations.push('consider lifestyle changes: weight loss, sleep position')
    recommendations.push('consult a doctor if symptoms persist')
  } else if (severity === 'moderate') {
    recommendations.push('moderate sleep apnea detected')
    recommendations.push('we recommend consulting a sleep specialist')
    recommendations.push('cpap therapy may be beneficial')
  } else {
    recommendations.push('severe sleep apnea detected')
    recommendations.push('please consult a sleep specialist soon')
    recommendations.push('treatment is important for your health')
  }

  return {
    sleepQuality: Math.round(sleepQuality),
    ahiScore: Math.round(ahiScore * 10) / 10,
    severity,
    events,
    recommendations
  }
}

export async function POST(request: Request) {
  try {
    const contentType = request.headers.get('content-type') || ''

    if (contentType.includes('multipart/form-data')) {
      const formData = await request.formData()
      const audioFile = formData.get('audio') || formData.get('file')

      if (!audioFile) {
        return NextResponse.json(
          { error: 'NO_AUDIO', message: 'no audio file provided' },
          { status: 400 }
        )
      }

      // simulate processing delay
      await new Promise(resolve => setTimeout(resolve, 2000))

      const result = generateMockAnalysis()

      return NextResponse.json({
        success: true,
        analysis: result,
        message: 'this is a demo analysis. deploy the python backend for real results.'
      })
    } else {
      return NextResponse.json({
        success: true,
        message: 'analysis api ready. send audio file as multipart/form-data.',
        demo: true
      })
    }
  } catch (error) {
    console.error('analysis error:', error)
    return NextResponse.json(
      { error: 'ANALYSIS_FAILED', message: 'failed to analyze audio' },
      { status: 500 }
    )
  }
}

export async function GET() {
  return NextResponse.json({
    status: 'ready',
    version: '1.0.0',
    demo: true,
    message: 'neendai analysis api. post audio for analysis.'
  })
}

export async function OPTIONS() {
  return new NextResponse(null, { status: 200 })
}
