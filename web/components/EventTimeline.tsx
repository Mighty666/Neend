'use client'

import { motion } from 'framer-motion'

interface Event {
  timestamp: number
  type: 'normal' | 'snoring' | 'hypopnea' | 'apnea'
  confidence: number
}

interface EventTimelineProps {
  events: Event[]
  maxEvents?: number
}

export function EventTimeline({ events, maxEvents = 50 }: EventTimelineProps) {
  const displayEvents = events.slice(-maxEvents)

  const eventColors = {
    normal: 'bg-green-500',
    snoring: 'bg-yellow-500',
    hypopnea: 'bg-orange-500',
    apnea: 'bg-red-500',
  }

  const eventHeights = {
    normal: 'h-4',
    snoring: 'h-6',
    hypopnea: 'h-8',
    apnea: 'h-10',
  }

  return (
    <div className="glass rounded-xl p-6">
      <h3 className="text-lg font-semibold mb-4">Event Timeline</h3>

      <div className="flex items-end gap-0.5 h-12">
        {displayEvents.map((event, i) => (
          <motion.div
            key={i}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            transition={{ delay: i * 0.02 }}
            className={`flex-1 ${eventColors[event.type]} ${eventHeights[event.type]} rounded-t-sm opacity-80 hover:opacity-100 transition-opacity cursor-pointer`}
            title={`${event.type} (${(event.confidence * 100).toFixed(0)}%)`}
          />
        ))}
      </div>

      {/* Legend */}
      <div className="flex justify-center gap-4 mt-4 text-xs text-gray-400">
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 bg-green-500 rounded-sm" />
          Normal
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 bg-yellow-500 rounded-sm" />
          Snoring
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 bg-orange-500 rounded-sm" />
          Hypopnea
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 bg-red-500 rounded-sm" />
          Apnea
        </div>
      </div>
    </div>
  )
}
