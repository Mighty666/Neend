'use client'

import { motion } from 'framer-motion'
import Link from 'next/link'
import { ArrowRightIcon, ShieldCheckIcon, BoltIcon, ChartBarIcon } from '@heroicons/react/24/outline'

export default function Home() {
  return (
    <main className="min-h-screen">
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 glass-dark">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="text-2xl font-bold"
          >
            <span className="text-gradient">Neend</span>
            <span className="text-white">AI</span>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex gap-4"
          >
            <Link href="/auth/signin" className="px-4 py-2 text-sm text-gray-300 hover:text-white transition-colors">
              Sign In
            </Link>
            <Link href="/auth/signup" className="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors">
              Get Started
            </Link>
          </motion.div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-6">
        <div className="max-w-7xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
              Know Your Breath.
              <br />
              <span className="text-gradient">Own Your Night.</span>
            </h1>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-10">
              AI-powered sleep apnea detection that watches while you rest.
              Clinical-grade analysis. Zero lab visits. Total peace of mind.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.6 }}
            className="flex flex-col sm:flex-row gap-4 justify-center"
          >
            <Link
              href="/auth/signup"
              className="group px-8 py-4 bg-blue-600 hover:bg-blue-500 rounded-xl text-lg font-semibold transition-all flex items-center justify-center gap-2"
            >
              Start Free Analysis
              <ArrowRightIcon className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              href="#features"
              className="px-8 py-4 glass hover:bg-white/10 rounded-xl text-lg font-semibold transition-colors"
            >
              See How It Works
            </Link>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
            className="mt-20 grid grid-cols-3 gap-8 max-w-3xl mx-auto"
          >
            {[
              { value: '95%', label: 'Accuracy' },
              { value: '<2s', label: 'Detection' },
              { value: '24/7', label: 'Monitoring' },
            ].map((stat, i) => (
              <div key={i} className="text-center">
                <div className="text-3xl md:text-4xl font-bold text-gradient">{stat.value}</div>
                <div className="text-sm text-gray-500 mt-1">{stat.label}</div>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Breathing Animation */}
      <section className="py-20 relative overflow-hidden">
        <motion.div
          animate={{ scale: [1, 1.1, 1], opacity: [0.3, 0.6, 0.3] }}
          transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
          className="absolute inset-0 flex items-center justify-center"
        >
          <div className="w-96 h-96 rounded-full bg-blue-500/20 blur-3xl" />
        </motion.div>
        <div className="relative z-10 text-center">
          <p className="text-gray-500 text-lg">Breathe. We're watching.</p>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <motion.h2
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-3xl md:text-4xl font-bold text-center mb-16"
          >
            Sleep Science, <span className="text-gradient">Simplified</span>
          </motion.h2>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: BoltIcon,
                title: 'Real-Time Detection',
                description: 'Instant classification of breathing patterns. Apnea, hypopnea, snoring—identified in seconds.',
              },
              {
                icon: ChartBarIcon,
                title: 'AHI Scoring',
                description: 'Clinical-grade Apnea-Hypopnea Index calculation with hourly breakdowns and trend analysis.',
              },
              {
                icon: ShieldCheckIcon,
                title: 'Privacy First',
                description: 'Your sleep data never leaves your device. Local AI processing. Zero cloud dependency.',
              },
            ].map((feature, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="glass p-8 rounded-2xl hover:bg-white/10 transition-colors group"
              >
                <feature.icon className="w-10 h-10 text-blue-400 mb-4 group-hover:scale-110 transition-transform" />
                <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
                <p className="text-gray-400">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-6">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          className="max-w-4xl mx-auto glass rounded-3xl p-12 text-center relative overflow-hidden"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-blue-600/20 to-cyan-600/20" />
          <div className="relative z-10">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Stop Guessing. Start Sleeping.
            </h2>
            <p className="text-gray-400 mb-8 max-w-xl mx-auto">
              80% of sleep apnea goes undiagnosed. Don't be a statistic.
              Get answers tonight.
            </p>
            <Link
              href="/auth/signup"
              className="inline-flex items-center gap-2 px-8 py-4 bg-white text-dark-950 rounded-xl text-lg font-semibold hover:bg-gray-100 transition-colors"
            >
              Analyze Your Sleep Free
              <ArrowRightIcon className="w-5 h-5" />
            </Link>
          </div>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 border-t border-white/5">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="text-gray-500 text-sm">
            © 2025 NeendAI. Sleep better. Live better.
          </div>
          <div className="flex gap-6 text-sm text-gray-500">
            <Link href="/privacy" className="hover:text-white transition-colors">Privacy</Link>
            <Link href="/terms" className="hover:text-white transition-colors">Terms</Link>
            <Link href="/contact" className="hover:text-white transition-colors">Contact</Link>
          </div>
        </div>
      </footer>
    </main>
  )
}
