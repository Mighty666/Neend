'use client'

import { motion } from 'framer-motion'
import { ArrowTrendingUpIcon, ArrowTrendingDownIcon } from '@heroicons/react/24/outline'

interface StatsCardProps {
  title: string
  value: string | number
  subtitle?: string
  trend?: {
    value: number
    label: string
    positive?: boolean
  }
  icon?: React.ReactNode
  color?: 'blue' | 'green' | 'yellow' | 'red' | 'purple'
  delay?: number
}

export function StatsCard({
  title,
  value,
  subtitle,
  trend,
  icon,
  color = 'blue',
  delay = 0,
}: StatsCardProps) {
  const colors = {
    blue: 'text-blue-400',
    green: 'text-green-400',
    yellow: 'text-yellow-400',
    red: 'text-red-400',
    purple: 'text-purple-400',
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5 }}
      className="glass rounded-xl p-5"
    >
      <div className="flex items-start justify-between mb-3">
        <p className="text-sm text-gray-400">{title}</p>
        {icon && <div className={colors[color]}>{icon}</div>}
      </div>

      <p className={`text-3xl font-bold ${colors[color]}`}>{value}</p>

      {subtitle && <p className="text-sm text-gray-500 mt-1">{subtitle}</p>}

      {trend && (
        <div className="flex items-center gap-1 mt-2 text-sm">
          {trend.positive !== undefined ? (
            trend.positive ? (
              <ArrowTrendingUpIcon className="w-4 h-4 text-green-400" />
            ) : (
              <ArrowTrendingDownIcon className="w-4 h-4 text-red-400" />
            )
          ) : null}
          <span className={trend.positive ? 'text-green-400' : 'text-red-400'}>
            {trend.value > 0 ? '+' : ''}{trend.value}
          </span>
          <span className="text-gray-500">{trend.label}</span>
        </div>
      )}
    </motion.div>
  )
}
