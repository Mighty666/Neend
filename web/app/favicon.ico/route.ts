import { NextResponse } from 'next/server'

// Return empty response for favicon to prevent 404 errors
export async function GET() {
  return new NextResponse(null, { status: 204 })
}

