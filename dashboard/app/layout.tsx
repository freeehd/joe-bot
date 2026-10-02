import "./globals.css";
import { Nav } from "@/components/Nav";
export const metadata = { title: "ARGUS — Joe Bot", description: "Risk-first quantitative trading operator console" };
export default function RootLayout({children}:{children:React.ReactNode}) { return <html lang="en"><body><Nav/><main className="main">{children}</main></body></html> }
