import type { Metadata } from 'next'
import './globals.css'
import { Toaster } from 'react-hot-toast'

export const metadata: Metadata = {
  title: 'NeendAI - Sleep Apnea Detection',
  description: 'AI-powered sleep analysis. Know your breath. Own your night.',
  icons: {
    icon: '/favicon.ico',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className="font-sans bg-dark-950 text-white antialiased">
        <Toaster
          position="top-center"
          toastOptions={{
            style: {
              background: '#1a1a1f',
              color: '#fff',
              border: '1px solid #2f416d',
            },
          }}
        />
        {children}
      </body>
    </html>
  )
}
