/** slugify converts a title to a URL slug.
 * BUG: edge dashes are not trimmed - "--Lead--" -> "-lead-". */
export function slugify(title: string): string {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-");
}

/** unslug converts a slug back to a title-cased string. */
export function unslug(slug: string): string {
  return slug
    .split("-")
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}
