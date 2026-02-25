import { Navigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";

interface Props {
  children: React.ReactNode;
}

/**
 * Wraps a route to redirect unauthenticated users to /login.
 * Shows nothing while the initial token validation is in flight.
 */
export function ProtectedRoute({ children }: Props) {
  const { isLoading, token } = useAuth();

  if (isLoading) {
    return (
      <div className="splash">
        <span className="splash__spinner" />
      </div>
    );
  }

  if (!token) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}
