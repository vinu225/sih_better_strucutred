/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: '#f4f8fc',
        brand: {
          blue: '#1d5cf0',
          green: '#1aa24b',
          'blue-tint': '#eaf1ff',
          'green-tint': '#e6f7ec',
          dark: '#0f1b33',
          muted: '#56637a',
          border: '#e2e9f3',
        }
      },
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'sans-serif'],
        display: ['"Instrument Serif"', 'Georgia', 'serif'],
      },
      boxShadow: {
        'soft': '0 2px 12px -2px rgba(15, 27, 51, 0.06), 0 1px 3px 0 rgba(15, 27, 51, 0.04)',
        'soft-hover': '0 8px 24px -4px rgba(29, 92, 240, 0.12), 0 2px 6px 0 rgba(15, 27, 51, 0.04)',
      },
      borderRadius: {
        'card': '16px',
      }
    },
  },
  plugins: [],
}
