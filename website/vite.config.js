import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const [owner, repo] = (process.env.GITHUB_REPOSITORY ?? '').split('/')
const inferredBase =
  repo && owner && repo.toLowerCase() !== `${owner.toLowerCase()}.github.io`
    ? `/${repo}/`
    : '/'

// Default to the repo path for GitHub Pages project sites, but allow overrides.
export default defineConfig({
  plugins: [react()],
  base: process.env.VITE_BASE_PATH ?? inferredBase,
})
