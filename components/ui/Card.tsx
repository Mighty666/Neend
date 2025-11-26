'use client'

import { forwardRef } from 'react'
import { motion } from 'framer-motion'
import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  animate?: boolean
  delay?: number
}

const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ className, animate = true, delay = 0, children, ...props }, ref) => {
    const content = (
      <div
        ref={ref}
        className={cn(
          'glass rounded-xl p-6',
          className
        )}
        {...props}
      >
        {children}
      </div>
    )

    if (animate) {
      return (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay, duration: 0.5 }}
        >
          {content}
        </motion.div>
      )
    }

    return content
  }
)

Card.displayName = 'Card'

export { Card }
