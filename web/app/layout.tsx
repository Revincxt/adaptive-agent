import type { Metadata, Viewport } from "next";
import { headers } from "next/headers";
import "./globals.css";

const pagesBasePath = process.env.GITHUB_PAGES_BASE_PATH ?? "/";
const sitesOrigin =
  process.env.SITE_ORIGIN ?? "https://adaptive-agent-lab.my20000806.chatgpt.site";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host");
  const forwardedProtocol = requestHeaders.get("x-forwarded-proto");
  const protocol =
    forwardedProtocol ?? (host?.startsWith("localhost") || host?.startsWith("127.0.0.1") ? "http" : "https");
  const fallbackOrigin = pagesBasePath === "/" ? sitesOrigin : "https://revincxt.github.io";
  const origin = host ? `${protocol}://${host}` : fallbackOrigin;

  return {
    metadataBase: new URL(pagesBasePath, `${origin}/`),
    title: "Adaptive Agent Lab — Multi-map Replay Explorer",
    description:
      "A multi-map replay explorer for inspecting planning, reinforcement-learning, replanning, and hybrid controllers in structured warehouse scenarios.",
    applicationName: "Adaptive Agent Lab",
    keywords: [
      "reinforcement learning",
      "automated planning",
      "warehouse robotics",
      "Dyna-Q",
      "DQN",
      "replanning",
    ],
    authors: [{ name: "Revincxt", url: "https://github.com/Revincxt" }],
    alternates: { canonical: "./" },
    icons: { icon: "./favicon.svg" },
    openGraph: {
      type: "website",
      url: "./",
      siteName: "Adaptive Agent Lab",
      title: "Adaptive Agent Lab — Multi-map Replay Explorer",
      description:
        "Inspect six controllers across four structured warehouse maps with synchronized trajectories, state, and recorded disruptions.",
      images: [
        {
          url: "./og.png",
          width: 1536,
          height: 1024,
          alt: "Four warehouse map topologies with recorded controller trajectories",
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title: "Adaptive Agent Lab — Multi-map Replay Explorer",
      description:
        "Four structured warehouse maps, six controllers, and synchronized recorded replays.",
      images: ["./og.png"],
    },
  };
}

export const viewport: Viewport = {
  colorScheme: "light",
  themeColor: "#f5f7f8",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
