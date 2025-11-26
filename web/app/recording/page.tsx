'use client'

import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { useRouter } from 'next/navigation'
import { StopIcon } from '@heroicons/react/24/solid'

export default function Recording() {
  const router = useRouter()
  const [duration, setDuration] = useState(0)
  const [events, setEvents] = useState({ normal: 0, snoring: 0, hypopnea: 0, apnea: 0 })
  const [currentEvent, setCurrentEvent] = useState<keyof typeof events>('normal')
  const [currentAHI, setCurrentAHI] = useState(0)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    // Duration timer
    const timer = setInterval(() => {
      setDuration(d => d + 1)
    }, 1000)

    // Simulate events for demo
    const eventTimer = setInterval(() => {
      const rand = Math.random()
      let event: keyof typeof events = 'normal'
      if (rand > 0.92) event = 'apnea'
      else if (rand > 0.82) event = 'hypopnea'
      else if (rand > 0.6) event = 'snoring'

      setCurrentEvent(event)
      setEvents(e => ({ ...e, [event]: e[event] + 1 }))

      // Update AHI
      setCurrentAHI(prev => {
        const hours = duration / 3600 || 0.001
        return (events.apnea + events.hypopnea + (event === 'apnea' || event === 'hypopnea' ? 1 : 0)) / hours
      })
    }, 3000)

    return () => {
      clearInterval(timer)
      clearInterval(eventTimer)
      wsRef.current?.close()
    }
  }, [duration, events])

  const formatDuration = (seconds: number) => {
    const h = Math.floor(seconds / 3600)
    const m = Math.floor((seconds % 3600) / 60)
    const s = seconds % 60
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
  }

  const stopRecording = () => {
    router.push('/dashboard')
  }

  const eventColors = {
    normal: 'bg-green-500',
    snoring: 'bg-yellow-500',
    hypopnea: 'bg-orange-500',
    apnea: 'bg-red-500',
  }

  return (
    <div className="min-h-screen bg-dark-950 flex items-center justify-center">
      <div className="text-center px-6 max-w-md w-full">
        {/* Breathing Animation */}
        <div className="relative w-48 h-48 mx-auto mb-8">
          <motion.div
            animate={{ scale: [1, 1.3, 1], opacity: [0.1, 0.3, 0.1] }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
            className="absolute inset-0 rounded-full bg-blue-500"
          />
          <motion.div
            animate={{ scale: [1, 1.2, 1], opacity: [0.2, 0.5, 0.2] }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut", delay: 0.2 }}
            className="absolute inset-6 rounded-full bg-blue-500"
          />
          <motion.div
            animate={{ scale: [1, 1.1, 1] }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
            className="absolute inset-12 rounded-full bg-blue-600 flex items-center justify-center"
          >
            <span className="text-4xl">🫁</span>
          </motion.div>
        </div>

        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
          <p className="text-gray-400 mb-2">Recording in Progress</p>
          <p className="text-5xl font-bold text-gradient mb-6 font-mono">{formatDuration(duration)}</p>
        </motion.div>

        {/* Current Event Badge */}
        <motion.div
          key={currentEvent}
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className={`inline-flex items-center gap-2 px-4 py-2 rounded-full mb-8 ${eventColors[currentEvent]}/20`}
        >
          <div className={`w-2 h-2 rounded-full ${eventColors[currentEvent]}`} />
          <span className="text-sm font-medium capitalize">{currentEvent}</span>
        </motion.div>

        {/* Event Counts */}
        <div className="grid grid-cols-4 gap-3 mb-8">
          {Object.entries(events).map(([key, value]) => (
            <div key={key} className="glass rounded-lg p-3">
              <p className="text-xl font-bold">{value}</p>
              <p className="text-xs text-gray-500 capitalize">{key}</p>
            </div>
          ))}
        </div>

        {/* Current AHI */}
        <div className="glass rounded-xl p-4 mb-8">
          <p className="text-sm text-gray-400 mb-1">Current AHI</p>
          <p className="text-3xl font-bold">{currentAHI.toFixed(1)}</p>
        </div>

        {/* Stop Button */}
        <motion.button
          whileTap={{ scale: 0.95 }}
          onClick={stopRecording}
          className="inline-flex items-center gap-2 px-8 py-4 bg-red-600 hover:bg-red-500 rounded-xl font-semibold transition-colors"
        >
          <StopIcon className="w-5 h-5" />
          Stop Recording
        </motion.button>
      </div>
    </div>
  )
}
