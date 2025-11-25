import { NextResponse } from 'next/server'

// Runtime configuration for Vercel serverless
export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const maxDuration = 30

// simple jwt-like token generation
// in production use a real jwt library but this works for demo
function generateToken(payload: object): string {
  // Use Buffer for Node.js compatibility (works in Vercel serverless)
  const header = Buffer.from(JSON.stringify({ alg: 'HS256', typ: 'JWT' })).toString('base64')
  const body = Buffer.from(JSON.stringify({ ...payload, exp: Date.now() + 86400000 })).toString('base64')
  const signature = Buffer.from(Math.random().toString(36).substring(2)).toString('base64')
  return `${header}.${body}.${signature}`
}

// using hardcoded demo user for serverless
// in production use vercel kv, planetscale, or similar
function getDemoUsers(): Record<string, { name: string; email: string; passwordHash: string }> {
  return {
    'demo@neendai.com': {
      name: 'Demo User',
      email: 'demo@neendai.com',
      passwordHash: 'demo123' // in production use bcrypt
    }
  }
}

export async function GET() {
  return NextResponse.json({
    status: 'ready',
    message: 'Sign in API endpoint. POST with email and password.'
  })
}

export async function POST(request: Request) {
  try {
    // Handle null/undefined request
    if (!request) {
      return NextResponse.json(
        { error: 'INVALID_REQUEST', message: 'Request is required' },
        { status: 400 }
      )
    }

    let body
    try {
      const text = await request.text()
      if (!text) {
        return NextResponse.json(
          { error: 'EMPTY_BODY', message: 'Request body is required' },
          { status: 400 }
        )
      }
      body = JSON.parse(text)
    } catch (parseError) {
      return NextResponse.json(
        { error: 'INVALID_JSON', message: 'Request body must be valid JSON' },
        { status: 400 }
      )
    }
    
    if (!body || typeof body !== 'object') {
      return NextResponse.json(
        { error: 'INVALID_BODY', message: 'Request body must be an object' },
        { status: 400 }
      )
    }
    
    const { email, password } = body

    // validate input
    if (!email || !password) {
      return NextResponse.json(
        { error: 'MISSING_FIELDS', message: 'email and password are required' },
        { status: 400 }
      )
    }

    const normalizedEmail = email.toLowerCase().trim()

    // check if user exists
    const DEMO_USERS = getDemoUsers()
    const user = DEMO_USERS[normalizedEmail]
    if (!user) {
      return NextResponse.json(
        { error: 'USER_NOT_FOUND', message: 'no account found with this email. try demo@neendai.com' },
        { status: 404 }
      )
    }

    // check password
    // todo: use bcrypt.compare in production
    if (user.passwordHash !== password) {
      return NextResponse.json(
        { error: 'INVALID_PASSWORD', message: 'incorrect password. hint: demo123' },
        { status: 401 }
      )
    }

    // generate token
    const token = generateToken({
      sub: user.email,
      name: user.name,
      iat: Date.now()
    })

    // return success
    return NextResponse.json(
      {
        success: true,
        user: { name: user.name, email: user.email },
        token
      },
      { status: 200 }
    )
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'unknown error'
    const errorStack = error instanceof Error ? error.stack : undefined
    console.error('signin error:', errorMessage, errorStack)
    return NextResponse.json(
      { 
        error: 'SERVER_ERROR', 
        message: 'something went wrong',
        details: process.env.NODE_ENV === 'development' ? errorMessage : undefined
      },
      { status: 500 }
    )
  }
}

// handle preflight for cors
export async function OPTIONS() {
  return new NextResponse(null, { status: 200 })
}
