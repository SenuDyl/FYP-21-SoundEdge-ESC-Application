export default function SelectField({
  label,
  value,
  onChange,
  items,
  loading,
  disabled,
  placeholderLoading,
  placeholderEmpty,
}) {
  const isEmpty = !items || items.length === 0;

  return (
    <div style={{ marginTop: 10, marginBottom: 10 }}>
      <div className="muted" style={{ marginBottom: 6 }}>
        {label}
      </div>

      <select
        className="select"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled || loading || isEmpty}
      >
        {isEmpty ? (
          <option value="">
            {loading ? placeholderLoading : placeholderEmpty}
          </option>
        ) : (
          items.map((it) => (
            <option key={it} value={it}>
              {it}
            </option>
          ))
        )}
      </select>
    </div>
  );
}
