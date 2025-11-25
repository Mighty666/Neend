import { NextResponse } from 'next/server'

// simple jwt-like token generation
// in production use a real jwt library but this works for demo
function generateToken(payload: object): string {
  const header = btoa(JSON.stringify({ alg: 'HS256', typ: 'JWT' }))
  const body = btoa(JSON.stringify({ ...payload, exp: Date.now() + 86400000 }))
  const signature = btoa(Math.random().toString(36).substring(2))
  return `${header}.${body}.${signature}`
}

// using hardcoded demo user for serverless
// in production use vercel kv, planetscale, or similar
const DEMO_USERS: Record<string, { name: string; email: string; passwordHash: string }> = {
  'demo@neendai.com': {
    name: 'Demo User',
    email: 'demo@neendai.com',
    passwordHash: 'demo123' // in production use bcrypt
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json()
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
  } catch (error) {
    console.error('signin error:', error)
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
