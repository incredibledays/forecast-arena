// Tailwind config for the ForecastArena Flask app.
// Mirrors the inline config that used to live in base.html when we
// depended on cdn.tailwindcss.com (the Play CDN — runtime JIT compiler,
// slow on every page load, especially over CN networks). We now
// pre-compile once into `static/css/tailwind.css`.
//
// Rebuild after changing template classes:
//   npx tailwindcss -i tailwind.input.css -o static/css/tailwind.css --minify
module.exports = {
  content: [
    "./templates/**/*.html",
    "./app.py",
  ],
  theme: {
    extend: {
      colors: {
        ink: "#0f172a",
        yes: "#16a34a",
        no:  "#dc2626",
      },
      fontFamily: {
        sans: ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "Inter", "sans-serif"],
      },
    },
  },
  safelist: [
    // Chart palette + status colors that show up only via dynamic strings.
    { pattern: /^(bg|text|border|ring)-(emerald|red|blue|orange|purple|cyan|lime|pink|teal|amber|indigo|rose)-(50|100|200|300|400|500|600|700|800|900)$/ },
    { pattern: /^(bg|text|border|ring)-(yes|no|ink)/ },
  ],
  plugins: [],
};
