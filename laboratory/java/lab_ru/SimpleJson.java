/*
 * SimpleJson.java — Minimal JSON parser/serializer (no external deps)
 */

package com.rmt.llm.lab.ru;

import java.util.*;
import java.util.regex.*;

public class SimpleJson {

    @SuppressWarnings("unchecked")
    public static Map<String, Object> parse(String json) {
        return (Map<String, Object>) parseValue(new JsonReader(json.trim()));
    }

    static class JsonReader {
        String s; int pos;
        JsonReader(String s) { this.s = s; }
        char peek() { return pos < s.length() ? s.charAt(pos) : '\0'; }
        char next() { return pos < s.length() ? s.charAt(pos++) : '\0'; }
        void skipWs() { while (pos < s.length() && Character.isWhitespace(s.charAt(pos))) pos++; }
    }

    static Object parseValue(JsonReader r) {
        r.skipWs();
        char c = r.peek();
        if (c == '{') return parseObject(r);
        if (c == '[') return parseArray(r);
        if (c == '"') return parseString(r);
        if (c == 't' || c == 'f') return parseBool(r);
        if (c == 'n') { r.next(); r.next(); r.next(); r.next(); return null; }
        return parseNumber(r);
    }

    static Map<String, Object> parseObject(JsonReader r) {
        Map<String, Object> m = new LinkedHashMap<>();
        r.next(); // skip {
        r.skipWs();
        if (r.peek() == '}') { r.next(); return m; }
        while (true) {
            r.skipWs();
            String key = parseString(r);
            r.skipWs();
            r.next(); // :
            Object val = parseValue(r);
            m.put(key, val);
            r.skipWs();
            char c = r.next();
            if (c == '}') break;
        }
        return m;
    }

    static List<Object> parseArray(JsonReader r) {
        List<Object> l = new ArrayList<>();
        r.next(); // skip [
        r.skipWs();
        if (r.peek() == ']') { r.next(); return l; }
        while (true) {
            l.add(parseValue(r));
            r.skipWs();
            char c = r.next();
            if (c == ']') break;
        }
        return l;
    }

    static String parseString(JsonReader r) {
        StringBuilder sb = new StringBuilder();
        r.next(); // skip "
        while (r.peek() != '"' && r.peek() != '\0') {
            char c = r.next();
            if (c == '\\') {
                char e = r.next();
                switch (e) {
                    case 'n': sb.append('\n'); break;
                    case 't': sb.append('\t'); break;
                    case 'r': sb.append('\r'); break;
                    case '"': sb.append('"'); break;
                    case '\\': sb.append('\\'); break;
                    case '/': sb.append('/'); break;
                    case 'u':
                        sb.append((char) Integer.parseInt(r.s.substring(r.pos, r.pos + 4), 16));
                        r.pos += 4; break;
                    default: sb.append(e);
                }
            } else sb.append(c);
        }
        r.next(); // skip "
        return sb.toString();
    }

    static Object parseBool(JsonReader r) {
        if (r.peek() == 't') { r.next(); r.next(); r.next(); r.next(); return true; }
        r.next(); r.next(); r.next(); r.next(); r.next(); return false;
    }

    static Number parseNumber(JsonReader r) {
        int start = r.pos;
        while (r.pos < r.s.length() && "-+0123456789.eE".indexOf(r.s.charAt(r.pos)) >= 0) r.pos++;
        String num = r.s.substring(start, r.pos);
        if (num.contains(".") || num.contains("e") || num.contains("E"))
            return Double.parseDouble(num);
        try { return Integer.parseInt(num); } catch (Exception e) { return Double.parseDouble(num); }
    }

    // Serializer
    public static String stringify(Object obj, int indent) {
        StringBuilder sb = new StringBuilder();
        writeValue(sb, obj, 0, indent);
        return sb.toString();
    }

    @SuppressWarnings("unchecked")
    static void writeValue(StringBuilder sb, Object v, int depth, int indent) {
        if (v == null) { sb.append("null"); return; }
        if (v instanceof String) { writeString(sb, (String) v); return; }
        if (v instanceof Boolean) { sb.append(v); return; }
        if (v instanceof Number) {
            Number n = (Number) v;
            if (n instanceof Double && (Double.isInfinite(n.doubleValue()) || Double.isNaN(n.doubleValue())))
                sb.append("null");
            else sb.append(n);
            return;
        }
        String pad = indent > 0 ? repeat(" ", depth * indent) : "";
        String padEnd = indent > 0 ? repeat(" ", (depth - 1) * indent) : "";
        String sep = indent > 0 ? ",\n" : ",";
        if (v instanceof Map) {
            Map<String, Object> m = (Map<String, Object>) v;
            if (m.isEmpty()) { sb.append("{}"); return; }
            sb.append("{");
            if (indent > 0) sb.append("\n");
            int i = 0;
            for (Map.Entry<String, Object> e : m.entrySet()) {
                if (i++ > 0) sb.append(sep);
                if (indent > 0) sb.append(pad);
                writeString(sb, e.getKey());
                sb.append(indent > 0 ? ": " : ":");
                writeValue(sb, e.getValue(), depth + 1, indent);
            }
            if (indent > 0) sb.append("\n").append(padEnd);
            sb.append("}");
        } else if (v instanceof List) {
            List<Object> l = (List<Object>) v;
            if (l.isEmpty()) { sb.append("[]"); return; }
            sb.append("[");
            if (indent > 0) sb.append("\n");
            int i = 0;
            for (Object item : l) {
                if (i++ > 0) sb.append(sep);
                if (indent > 0) sb.append(pad);
                writeValue(sb, item, depth + 1, indent);
            }
            if (indent > 0) sb.append("\n").append(padEnd);
            sb.append("]");
        } else {
            writeString(sb, String.valueOf(v));
        }
    }

    static void writeString(StringBuilder sb, String s) {
        sb.append('"');
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"': sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\n': sb.append("\\n"); break;
                case '\t': sb.append("\\t"); break;
                case '\r': sb.append("\\r"); break;
                default:
                    if (c < 0x20) sb.append(String.format("\\u%04x", (int) c));
                    else sb.append(c);
            }
        }
        sb.append('"');
    }

    static String repeat(String s, int n) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) sb.append(s);
        return sb.toString();
    }
}
