import type { ReactNode } from "react";

export function Card({
  id,
  number,
  title,
  children,
}: {
  id: string;
  number: string;
  title: string;
  children: ReactNode;
}) {
  return (
    <section className="card" id={id} aria-labelledby={`${id}-title`}>
      <div className="section-header">
        <span className="badge">{number}</span>
        <h2 className="section-title" id={`${id}-title`}>
          {title}
        </h2>
      </div>
      {children}
    </section>
  );
}

export function Kpi({ value, label, note }: { value: string; label: string; note?: string }) {
  return (
    <div className="kpi">
      <p className="kpi-value">{value}</p>
      <p className="kpi-label">
        {label}
        {note && <small>{note}</small>}
      </p>
    </div>
  );
}
