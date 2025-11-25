import { NextResponse } from 'next/server'

// simple token generation (same as signin)
function generateToken(payload: object): string {
  const header = btoa(JSON.stringify({ alg: 'HS256', typ: 'JWT' }))
  const body = btoa(JSON.stringify({ ...payload, exp: Date.now() + 86400000 }))
  const signature = btoa(Math.random().toString(36).substring(2))
  return `${header}.${body}.${signature}`
}

// simulated database - in production use real db
// note: this resets on each deployment but works for demo
const registeredEmails = new Set(['demo@neendai.com'])

export async function POST(request: Request) {
  try {
    const body = await request.json()
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
  } catch (error) {
    console.error('signup error:', error)
    return NextResponse.json(
      { error: 'SERVER_ERROR', message: 'something went wrong' },
      { status: 500 }
    )
  }
}

// handle preflight for cors
export async function OPTIONS() {
  return new NextResponse(null, { status: 200 })
}
