import { NextResponse } from 'next/server'

// Runtime configuration for Vercel serverless
export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const maxDuration = 30

// simple token generation (same as signin)
function generateToken(payload: object): string {
  // Use Buffer for Node.js compatibility (works in Vercel serverless)
  const header = Buffer.from(JSON.stringify({ alg: 'HS256', typ: 'JWT' })).toString('base64')
  const body = Buffer.from(JSON.stringify({ ...payload, exp: Date.now() + 86400000 })).toString('base64')
  const signature = Buffer.from(Math.random().toString(36).substring(2)).toString('base64')
  return `${header}.${body}.${signature}`
}

// simulated database - in production use real db
// note: this resets on each deployment but works for demo
const registeredEmails = new Set(['demo@neendai.com'])

export async function GET() {
  return NextResponse.json({
    status: 'ready',
    message: 'Sign up API endpoint. POST with name, email, and password.'
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
      // Use request.json() directly - it's the proper way for JSON in Next.js
      body = await request.json()
    } catch (parseError) {
      // If json() fails, try to get more details
      const errorMsg = parseError instanceof Error ? parseError.message : 'Unknown error'
      console.error('JSON parse error:', errorMsg)
      return NextResponse.json(
        { error: 'INVALID_JSON', message: 'Request body must be valid JSON', details: errorMsg },
        { status: 400 }
      )
    }
    
    if (!body || typeof body !== 'object') {
      return NextResponse.json(
        { error: 'INVALID_BODY', message: 'Request body must be an object' },
        { status: 400 }
      )
    }
    
    const { name, email, password } = body

    // validate input
    if (!name || !email || !password) {
      return NextResponse.json(
        { error: 'MISSING_FIELDS', message: 'all fields are required' },
        { status: 400 }
      )
    }

    // validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    if (!emailRegex.test(email)) {
      return NextResponse.json(
        { error: 'INVALID_EMAIL', message: 'please enter a valid email' },
        { status: 400 }
      )
    }

    // validate password strength
    if (password.length < 6) {
      return NextResponse.json(
        { error: 'WEAK_PASSWORD', message: 'password must be at least 6 characters' },
        { status: 400 }
      )
    }

    const normalizedEmail = email.toLowerCase().trim()

    // check if email already registered
    if (registeredEmails.has(normalizedEmail)) {
      return NextResponse.json(
        { error: 'EMAIL_EXISTS', message: 'this email is already registered. try signing in instead.' },
        { status: 409 }
      )
    }

    // register user (in production, save to database with hashed password)
    registeredEmails.add(normalizedEmail)

    // generate token
    const token = generateToken({
      sub: normalizedEmail,
      name: name.trim(),
      iat: Date.now()
    })

    // return success
    return NextResponse.json(
      {
        success: true,
        user: { name: name.trim(), email: normalizedEmail },
        token
      },
      { status: 201 }
    )
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'unknown error'
    const errorStack = error instanceof Error ? error.stack : undefined
    console.error('signup error:', errorMessage, errorStack)
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
