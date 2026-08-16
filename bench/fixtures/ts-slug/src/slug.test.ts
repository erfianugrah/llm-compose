import { test, expect } from "bun:test";
import { unslug } from "./slug";

test("unslug basic", () => {
  expect(unslug("hello-world")).toBe("Hello World");
});

test("unslug with empties", () => {
  expect(unslug("--a-b--")).toBe("A B");
});
