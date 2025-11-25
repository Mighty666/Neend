'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { EnvelopeIcon, LockClosedIcon, ExclamationCircleIcon } from '@heroicons/react/24/outline'
import toast from 'react-hot-toast'

export default function SignIn() {
  const router = useRouter()
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  })
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [isLoading, setIsLoading] = useState(false)

  const validateForm = () => {
    const newErrors: Record<string, string> = {}

    if (!formData.email.trim()) {
      newErrors.email = 'Email is required'
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      newErrors.email = 'Invalid email format'
    }

    if (!formData.password) {
      newErrors.password = 'Password is required'
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleGuestLogin = async () => {
    setFormData({
      email: 'demo@neendai.com',
      password: 'demo123'
    })
    setIsLoading(true)

    try {
      const response = await fetch('/api/auth/signin', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: 'demo@neendai.com',
          password: 'demo123'
        })
      })

      const data = await response.json()

      if (!response.ok) {
        toast.error(data.message || 'Something went wrong')
        return
      }

      toast.success('Welcome, Guest!')
      router.push('/dashboard')

    } catch (error) {
      toast.error('Network error. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!validateForm()) return

    setIsLoading(true)

    try {
      const response = await fetch('/api/auth/signin', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      })

      const data = await response.json()

      if (!response.ok) {
        if (data.error === 'USER_NOT_FOUND') {
          setErrors({ email: 'No account found with this email. Try signing up instead.' })
          toast.error('Account not found')
        } else if (data.error === 'INVALID_PASSWORD') {
          setErrors({ password: 'Incorrect password' })
          toast.error('Invalid credentials')
        } else {
          toast.error(data.message || 'Something went wrong')
        }
        return
      }

      toast.success('Welcome back!')
      router.push('/dashboard')

    } catch (error) {
      toast.error('Network error. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="min-h-screen flex items-center justify-center px-6 py-12">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md"
      >
        {/* Logo */}
        <Link href="/" className="block text-center mb-8">
          <span className="text-3xl font-bold">
            <span className="text-gradient">Neend</span>
            <span className="text-white">AI</span>
          </span>
        </Link>

        {/* Card */}
        <div className="glass rounded-2xl p-8">
          <h1 className="text-2xl font-bold text-center mb-2">Welcome Back</h1>
          <p className="text-gray-400 text-center mb-8">Sign in to view your sleep data</p>

          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Email */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Email</label>
              <div className="relative">
                <EnvelopeIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
                <input
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  className={`w-full bg-dark-900 border ${errors.email ? 'border-red-500' : 'border-white/10'} rounded-lg py-3 pl-10 pr-4 text-white placeholder-gray-500 focus:border-blue-500 transition-colors`}
                  placeholder="you@example.com"
                />
              </div>
              {errors.email && (
                <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-1 text-sm text-red-400 flex items-center gap-1">
                  <ExclamationCircleIcon className="w-4 h-4" />
                  {errors.email}
                </motion.p>
              )}
            </div>

            {/* Password */}
            <div>
              <div className="flex justify-between items-center mb-2">
                <label className="text-sm font-medium text-gray-300">Password</label>
                <Link href="/auth/forgot-password" className="text-xs text-blue-400 hover:text-blue-300 transition-colors">
                  Forgot password?
                </Link>
              </div>
              <div className="relative">
                <LockClosedIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
                <input
                  type="password"
                  value={formData.password}
                  onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                  className={`w-full bg-dark-900 border ${errors.password ? 'border-red-500' : 'border-white/10'} rounded-lg py-3 pl-10 pr-4 text-white placeholder-gray-500 focus:border-blue-500 transition-colors`}
                  placeholder="••••••••"
                />
              </div>
              {errors.password && (
                <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-1 text-sm text-red-400 flex items-center gap-1">
                  <ExclamationCircleIcon className="w-4 h-4" />
                  {errors.password}
                </motion.p>
              )}
            </div>

            {/* Submit */}
            <motion.button
              type="submit"
              disabled={isLoading}
              whileTap={{ scale: 0.98 }}
              className="w-full py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-blue-600/50 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                'Sign In'
              )}
            </motion.button>
          </form>

          {/* Guest Login */}
          <motion.button
            type="button"
            onClick={handleGuestLogin}
            disabled={isLoading}
            whileTap={{ scale: 0.98 }}
            className="w-full mt-3 py-3 bg-white/5 hover:bg-white/10 disabled:bg-white/5 border border-white/10 rounded-lg font-semibold transition-colors"
          >
            Continue as Guest
          </motion.button>

          <p className="mt-6 text-center text-gray-400 text-sm">
            Don't have an account?{' '}
            <Link href="/auth/signup" className="text-blue-400 hover:text-blue-300 transition-colors">
              Sign up
            </Link>
          </p>
        </div>
      </motion.div>
    </main>
  )
}
