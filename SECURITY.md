# Security Policy

## Supported version

Security fixes are currently provided for the latest public-beta branch only. This is research
software and carries no clinical or production-service support commitment.

## Reporting a vulnerability

Do not open a public issue containing credentials, exploitable generated code, private datasets,
or sensitive task archives. Use GitHub's private security-advisory mechanism after the repository
is published. Until then, contact the repository owner through a private channel stated on the
owner's GitHub profile.

Include the affected commit, reproduction steps, impact, and whether the issue crosses the host,
Docker, dataset, provider-key, test-seal, or archive trust boundary. Remove all real API keys and
patient-related data from reports.

## Security boundaries

- LLM-generated experiment code is untrusted and must run in the network-disabled, non-root
  container with no provider credentials.
- Pickle and NumPy object serialization from untrusted sources must not be loaded.
- Imported task archives and datasets are untrusted until their schema, path ownership, and hashes
  are validated.
- The application is not approved for clinical diagnosis or autonomous medical decision-making.
