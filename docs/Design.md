# Design.md — Visual Design System

## Theme Direction
Dark, technical, "privacy/cryptography" mood — should feel like a security/crypto console, not a consumer fintech app. Calm and precise rather than flashy. Think: a quiet dashboard where a proof is being generated in the background.

## Color Palette

| Role | Color | Hex | Usage |
|---|---|---|---|
| Background (base) | Near-black navy | `#0B0E14` | App background |
| Surface | Dark slate | `#151A24` | Cards, panels |
| Primary accent | Teal/emerald | `#2DD4BF` | Local/on-device elements (matches "data never leaves device" concept) |
| Secondary accent | Indigo/purple | `#7C6CF5` | Verifier / proof elements |
| Warning/error | Coral | `#F87171` | Errors, invalid proof states |
| Success | Soft green | `#4ADE80` | Verified / valid proof states |
| Text primary | Off-white | `#E5E7EB` | Main text |
| Text secondary | Muted gray | `#8B93A7` | Labels, helper text |
| Border/divider | Subtle gray | `#232838` | Card borders, dividers |

**Color logic:** teal = local/on-device (private) actions; indigo/purple = verifier/network actions. This mirrors the architecture diagram (teal user-device box, coral/verifier box) and reinforces the core concept visually — the user should be able to tell "this is happening on my device" vs. "this just left my device" at a glance.

## Typography

| Use | Font | Notes |
|---|---|---|
| Headings / UI | Inter or Space Grotesk | Clean, modern geometric sans |
| Body text | Inter | Readable at small sizes |
| Proofs, hashes, JSON, code | JetBrains Mono or Fira Code | Monospace for anything cryptographic — proof hex, circuit output, keys |

**Sizing scale:** 12 / 14 / 16 / 20 / 28 / 36 px, with 1.4–1.6 line-height for body text.

## UI Patterns
- **Status indicators as a pipeline**, not a spinner: show discrete steps (Local Inference → Proof Generation → Verification) with checkmarks/progress per step — reinforces that these are distinct, real cryptographic operations, not a black box.
- **Proof output shown in a monospace "console" card** (truncated hex/hash with a copy button) — makes the ZK proof feel tangible rather than abstract.
- **Never show raw input data anywhere in a "sent" or "transmitted" context** — the UI itself should visually reinforce the privacy guarantee (e.g., a small lock/shield icon or "stays on this device" label next to the input form).
- Cards with subtle borders (`#232838`) and soft shadows, low-contrast surfaces, generous padding — avoid clutter.
- Rounded corners: 8–12px for cards, 6px for buttons/inputs.

## Motion
- Subtle only — fade/slide transitions between pipeline steps (200–300ms ease-out). No heavy animation; this is a technical/trust-focused interface, not a marketing site.

## Accessibility
- Maintain WCAG AA contrast for text on dark backgrounds (off-white `#E5E7EB` on `#0B0E14`/`#151A24` passes comfortably).
- Don't rely on color alone for valid/invalid states — pair with icons/text (✓ Verified / ✕ Invalid).
