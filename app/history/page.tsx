'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'
import { format } from 'date-fns'
import { ChevronRightIcon, CalendarIcon, ClockIcon } from '@heroicons/react/24/outline'

// Mock data
const sessions = [
  { id: '1', date: new Date('2025-11-22'), duration: 7.5, ahi: 12.4, severity: 'Mild', events: 47 },
  { id: '2', date: new Date('2025-11-21'), duration: 6.8, ahi: 14.1, severity: 'Mild', events: 52 },
  { id: '3', date: new Date('2025-11-20'), duration: 8.2, ahi: 8.3, severity: 'Mild', events: 38 },
  { id: '4', date: new Date('2025-11-19'), duration: 7.1, ahi: 22.7, severity: 'Moderate', events: 89 },
  { id: '5', date: new Date('2025-11-18'), duration: 6.5, ahi: 18.9, severity: 'Moderate', events: 71 },
]

export default function History() {
  const [selectedSession, setSelectedSession] = useState<string | null>(null)

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'Normal': return 'text-green-400 bg-green-400/10'
      case 'Mild': return 'text-yellow-400 bg-yellow-400/10'
      case 'Moderate': return 'text-orange-400 bg-orange-400/10'
      case 'Severe': return 'text-red-400 bg-red-400/10'
      default: return 'text-gray-400 bg-gray-400/10'
    }
  }

  return (
    <div className="min-h-screen bg-dark-950">
      {/* Header */}
      <header className="glass-dark sticky top-0 z-50">
        <div className="max-w-4xl mx-auto px-6 py-4 flex justify-between items-center">
          <Link href="/dashboard" className="text-xl font-bold">
            <span className="text-gradient">Neend</span>
            <span className="text-white">AI</span>
          </Link>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-2xl font-bold mb-1">Sleep History</h1>
          <p className="text-gray-400">View and compare your past sleep sessions</p>
        </motion.div>

        {/* Sessions List */}
        <div className="space-y-4">
          {sessions.map((session, i) => (
            <motion.div
              key={session.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
            >
              <Link href={`/reports/${session.id}`}>
                <div className="glass rounded-xl p-5 hover:bg-white/10 transition-colors cursor-pointer group">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-12 rounded-lg bg-blue-500/20 flex items-center justify-center">
                        <CalendarIcon className="w-6 h-6 text-blue-400" />
                      </div>
                      <div>
                        <p className="font-semibold">{format(session.date, 'EEEE, MMM d')}</p>
                        <div className="flex items-center gap-3 text-sm text-gray-400">
                          <span className="flex items-center gap-1">
                            <ClockIcon className="w-4 h-4" />
                            {session.duration}h
                          </span>
                          <span>{session.events} events</span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <p className="text-2xl font-bold">{session.ahi.toFixed(1)}</p>
                        <span className={`text-xs px-2 py-0.5 rounded-full ${getSeverityColor(session.severity)}`}>
                          {session.severity}
                        </span>
                      </div>
                      <ChevronRightIcon className="w-5 h-5 text-gray-500 group-hover:text-white transition-colors" />
                    </div>
                  </div>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>

        {/* Weekly Average */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-8 glass rounded-xl p-6"
        >
          <h3 className="text-lg font-semibold mb-4">Weekly Summary</h3>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-gray-400">Avg AHI</p>
              <p className="text-2xl font-bold">15.3</p>
            </div>
            <div>
              <p className="text-sm text-gray-400">Avg Duration</p>
              <p className="text-2xl font-bold">7.2h</p>
            </div>
            <div>
              <p className="text-sm text-gray-400">Total Events</p>
              <p className="text-2xl font-bold">297</p>
            </div>
          </div>
        </motion.div>
      </main>
    </div>
  )
}
