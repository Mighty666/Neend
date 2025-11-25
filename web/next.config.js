/** @type {import('next').NextConfig} */

// next config for vercel deployment
// had to add a bunch of stuff to make it work properly

const nextConfig = {
  reactStrictMode: true,

  // serverless function configuration
  experimental: {
    serverActions: {
      bodySizeLimit: '10mb',
    },
  },

  // image optimization settings
  images: {
    domains: ['localhost'],
    unoptimized: process.env.NODE_ENV === 'development',
  },

  // environment variables that are safe to expose to browser
  env: {
    NEXT_PUBLIC_APP_NAME: 'NeendAI',
    NEXT_PUBLIC_APP_VERSION: '1.0.0',
  },

  // headers for cors and security
  async headers() {
    return [
      {
        source: '/api/:path*',
        headers: [
          { key: 'Access-Control-Allow-Credentials', value: 'true' },
          { key: 'Access-Control-Allow-Origin', value: '*' },
          { key: 'Access-Control-Allow-Methods', value: 'GET,DELETE,PATCH,POST,PUT' },
          { key: 'Access-Control-Allow-Headers', value: 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version, Authorization' },
        ],
      },
    ]
  },

  // redirects
  async redirects() {
    return [
      {
        source: '/home',
        destination: '/',
        permanent: true,
      },
    ]
  },
}

module.exports = nextConfig
