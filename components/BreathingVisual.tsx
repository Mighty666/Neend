'use client'

import { motion } from 'framer-motion'

interface BreathingVisualProps {
  isActive?: boolean
  event?: 'normal' | 'snoring' | 'hypopnea' | 'apnea'
  size?: 'sm' | 'md' | 'lg'
}

export function BreathingVisual({ isActive = true, event = 'normal', size = 'md' }: BreathingVisualProps) {
  const sizeMap = {
    sm: { outer: 'w-24 h-24', middle: 'inset-3', inner: 'inset-6' },
    md: { outer: 'w-48 h-48', middle: 'inset-6', inner: 'inset-12' },
    lg: { outer: 'w-64 h-64', middle: 'inset-8', inner: 'inset-16' },
  }

  const eventColors = {
    normal: 'bg-blue-500',
    snoring: 'bg-yellow-500',
    hypopnea: 'bg-orange-500',
    apnea: 'bg-red-500',
  }

  const duration = event === 'apnea' ? 8 : 4

  return (
    <div className={`relative ${sizeMap[size].outer}`}>
      {/* Outer ring */}
      <motion.div
        animate={isActive ? {
          scale: [1, 1.3, 1],
          opacity: [0.1, 0.3, 0.1]
        } : {}}
        transition={{
          duration,
          repeat: Infinity,
          ease: "easeInOut"
        }}
        className={`absolute inset-0 rounded-full ${eventColors[event]}`}
      />

      {/* Middle ring */}
      <motion.div
        animate={isActive ? {
          scale: [1, 1.2, 1],
          opacity: [0.2, 0.5, 0.2]
        } : {}}
        transition={{
          duration,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 0.2
        }}
        className={`absolute ${sizeMap[size].middle} rounded-full ${eventColors[event]}`}
      />

      {/* Inner circle */}
      <motion.div
        animate={isActive ? { scale: [1, 1.1, 1] } : {}}
        transition={{
          duration,
          repeat: Infinity,
          ease: "easeInOut"
        }}
        className={`absolute ${sizeMap[size].inner} rounded-full ${eventColors[event]} opacity-60 flex items-center justify-center`}
      >
        <span className={size === 'sm' ? 'text-xl' : size === 'md' ? 'text-4xl' : 'text-5xl'}>
          🫁
        </span>
      </motion.div>
    </div>
  )
}
