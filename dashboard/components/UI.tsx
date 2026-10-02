import { ReactNode } from "react";
export function Panel({title, children, aside}:{title:string,children:ReactNode,aside?:ReactNode}) { return <section className="panel"><header><h2>{title}</h2>{aside}</header>{children}</section> }
export function Metric({label,value,sub}:{label:string,value:string|number,sub?:string}) { return <div className="metric"><span>{label}</span><strong>{value}</strong>{sub && <small>{sub}</small>}</div> }
export function Status({value}:{value:string}) { const key=value.toLowerCase().replaceAll(" ","-"); return <span className={`status ${key}`}>{value}</span> }
export function Empty({children}:{children:string}) { return <div className="empty">{children}</div> }
