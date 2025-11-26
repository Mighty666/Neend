import { NextResponse } from 'next/server'

// Runtime configuration for Vercel serverless
export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const maxDuration = 30

// Minimal test route to verify serverless functions work
export async function GET() {
  try {
    return NextResponse.json({
      status: 'ok',
      message: 'Serverless function is working',
      timestamp: new Date().toISOString(),
      nodeVersion: process.version
    })
  } catch (error) {
    return NextResponse.json(
      { error: 'Test failed', message: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    )
  }
}

export async function POST() {
  return NextResponse.json({ message: 'POST method works' })
}

