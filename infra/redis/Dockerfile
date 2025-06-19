FROM redis:7-alpine

# Expose Redis port
EXPOSE 6379

# Start Redis server with password from environment
CMD ["sh", "-c", "redis-server --requirepass \"$REDIS_PASSWORD\""] 